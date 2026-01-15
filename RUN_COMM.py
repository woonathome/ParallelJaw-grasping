import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import time
import torch
from PIL import Image
import logging
from hydra import initialize, compose
import trimesh
import numpy as np
from hydra.utils import instantiate
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as transforms
import cv2
import imageio
import distinctipy
import gorilla
import importlib
import json, socket
from skimage.feature import canny
from skimage.morphology import binary_dilation
from scipy.spatial.transform import Rotation
from Instance_Segmentation_Model.segment_anything.utils.amg import rle_to_mask
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from Instance_Segmentation_Model.model.loss import Similarity
from Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# set level logging
logging.basicConfig(level=logging.INFO)


sys.path.append(os.path.join(os.getcwd(), 'Pose_Estimation_Model', 'provider'))
sys.path.append(os.path.join(os.getcwd(), 'Pose_Estimation_Model', 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'Pose_Estimation_Model', 'model'))
sys.path.append(os.path.join(os.getcwd(), 'Pose_Estimation_Model', 'model', 'pointnet2'))

from Pose_Estimation_Model.utils.data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from Pose_Estimation_Model.utils.draw_utils import draw_detections
import pycocotools.mask as cocomask
import trimesh

import Grasping_Face.grasping as gf
import Grasping_Face.visualize_grasping as gfv

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

def load_json(path):
    with open(path, "r") as f:
        info = json.load(f)
    return info

# torch.cuda.empty_cache()

############### Settings ###############
SCENE_DATE = time.strftime("%Y%m%d_%H%M%S")
SCENE_ID  = '0'
object_id = 'logitech_c930e_m'   # 'logitech_c930e_m' 'object_100'

# OPE model settings
segmentor_model  = 'fastsam' # 'sam' , 'fastsam'

template_dir = f'./models_target/templates/{object_id}' 
cad_path     = f'./models_target/models_cad/custom/{object_id}.obj'
output_dir   =  './RUN_result/output'
cam_path     = f'./RUN_result/test_rgbd/{time.strftime("%Y%m%d")}/scene_camera.json'
seg_path     = f'{output_dir}/detection_ism_{SCENE_DATE}_{SCENE_ID}_{object_id}.json'
stability_score_thresh = 0.97
det_score_thresh = 0.2
gpus = 0

# # communication settings
# comm = False # whether to make server on this computer
# HOST = '192.168.0.30'
# PORT = 9900

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############### Settings ###############

##### Instance Segmentation #####
def ISM_visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def ISM_visualize_all(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    colors = distinctipy.get_colors(len(detections))
    alpha = 0.1

    for idx, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = det["category_id"]
        temp_id = obj_id - 1  # for color indexing

        # 색상 지정
        r = int(255 * colors[temp_id % len(colors)][0])
        g = int(255 * colors[temp_id % len(colors)][1])
        b = int(255 * colors[temp_id % len(colors)][2])

        # 마스크 영역 색칠
        img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
        img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
        img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]

        # edge 강조
        img[edge, :] = 255

    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)

    # concat side by side in PIL
    concat = Image.new('RGB', (rgb.size[0] + prediction.size[0], rgb.size[1]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (rgb.size[0], 0))
    return concat

def ISM_batch_input_data(depth_path = '', cam_path = cam_path, device = device, scene_id = SCENE_ID):
    batch = {}
    cam_info = load_json(cam_path)[scene_id]
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def ISM_load(segmentor_model = segmentor_model, output_dir = output_dir, template_dir = template_dir, stability_score_thresh = stability_score_thresh):

    with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)

    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        

    logging.info("Initializing template")
    # template_dir = os.path.join(output_dir, 'templates')
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))

    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data

    torch.cuda.empty_cache()

    return model

def ISM_run_save(ISMmodel = None, cad_path = cad_path, rgb_path = '', depth_path = '', cam_path = cam_path, save_img = True, tag='default'):
    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    detections = ISMmodel.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = ISMmodel.descriptor_model.forward(np.array(rgb), detections)

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = ISMmodel.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= ISMmodel.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    # compute the geometric score
    batch = ISM_batch_input_data(depth_path, cam_path, device, SCENE_ID)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    ISMmodel.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    ISMmodel.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)

    image_uv = ISMmodel.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = ISMmodel.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=ISMmodel.visible_thred
        )

    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   

    # save ISM result
    detections.to_numpy()
    save_path = f"{output_dir}/detection_ism_{tag}_{object_id}"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])

    # --- [수정] ISM Score 기준 정렬, 상위 10개만 선택 ---
    detections_sorted = sorted(detections, key=lambda d: d['score'], reverse=True)
    detections_top10 = detections_sorted[:10]
    save_json_bop23(save_path+".json", detections_top10)

    if save_img:
        # 시각화 시에도 Top 10 (혹은 정렬된 전체)을 사용
        vis_img = ISM_visualize_all(rgb, detections_top10, f"{output_dir}/vis_ism_{tag}_{object_id}.png")
        vis_img.save(f"{output_dir}/vis_ism_{tag}_{object_id}.png")

    torch.cuda.empty_cache()

    return detections_top10

    # --- [수정전] ISM Score 기준으로 정렬하고 상위 10개만 선택 ---
    # save_json_bop23(save_path+".json", detections)

    # if save_img:
    #     vis_img = ISM_visualize_all(rgb, detections, f"{output_dir}/vis_ism_{tag}_{object_id}.png")
    #     vis_img.save(f"{output_dir}/vis_ism_{tag}_{object_id}.png")

    # torch.cuda.empty_cache()

    # return detections

##### Pose Estimation #####
def PEM_visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def _get_template(path, cfg, tem_index=1):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz

def get_templates(path, cfg):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
    return all_tem, all_tem_pts, all_tem_choose

def get_test_data(rgb_path = '', depth_path = '', cam_path = cam_path, cad_path = cad_path, seg_path = seg_path, det_score_thresh = det_score_thresh, cfg = None, scene_id = SCENE_ID, N_INSTANCE = 15): # RTX 5070 laptop GPU 메모리 기준: N_INSTANCE=30
    dets = []
    with open(seg_path) as f:
        dets_ = json.load(f) # keys: SCENE_ID, image_id, category_id, bbox, score, segmentation
    for det in dets_:
        if det['score'] > det_score_thresh:
            dets.append(det)
    del dets_

    cam_info = json.load(open(cam_path))[scene_id]
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    whole_image = load_im(rgb_path).astype(np.uint8)
    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    whole_depth = load_im(depth_path).astype(np.float32) * cam_info['depth_scale'] / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    for inst in dets:
        seg = inst['segmentation']
        score = inst['score']

        # mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # rgb
        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)


    ret_dict = {}
    ############ GPU OOM 에러 대응 ############
    # ret_dict['pts'] = torch.stack(all_cloud).cuda()
    # ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    # ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    # ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    # ninstance = ret_dict['pts'].size(0)
    # ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    # ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()

    if len(all_cloud) < N_INSTANCE:
        N_INSTANCE = len(all_cloud)

    ret_dict['pts'] = torch.stack(all_cloud[:N_INSTANCE]).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb[:N_INSTANCE]).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose[:N_INSTANCE]).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score[:N_INSTANCE]).cuda()

    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(N_INSTANCE, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(N_INSTANCE, 1, 1).cuda()

    all_dets = all_dets[:N_INSTANCE]
    ############ GPU OOM 에러 대응 ############

    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets

def PEM_init(gpus = gpus, output_dir = output_dir, template_dir = template_dir, cad_path = cad_path, rgb_path = '', depth_path = '', cam_path = cam_path, seg_path = seg_path, det_score_thresh = det_score_thresh):
    # exp_name = model_name + '_' + \
    #     osp.splitext(config.split("/")[-1])[0] + '_id' + str(exp_id)
    # log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile("Pose_Estimation_Model/config/base.yaml")
    # cfg.exp_name = exp_name
    cfg.gpus     = gpus
    cfg.model_name = 'pose_estimation_model'
    # cfg.log_dir  = log_dir
    cfg.test_iter = 600000

    cfg.output_dir = output_dir
    cfg.template_dir = template_dir
    cfg.cad_path = cad_path
    cfg.rgb_path = rgb_path
    cfg.depth_path = depth_path
    cfg.cam_path = cam_path
    cfg.seg_path = seg_path

    cfg.n_sample_observed_point = 512  # 2048
    cfg.n_sample_model_point = 256     # 1024
    cfg.n_sample_template_point = 500  # 5000

    cfg.det_score_thresh = det_score_thresh
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    return cfg

def PEM_load(cfg = None):
    # model
    print("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    model = model.cuda()
    model.eval()
    # checkpoint = os.path.join(os.path.dirname((os.getcwd())), 'Pose_Estimation_Model', 'checkpoints', 'sam-6d-pem-base.pth')
    checkpoint = os.path.join('./Pose_Estimation_Model', 'checkpoints', 'sam-6d-pem-base.pth')
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    return model

def PEM_run_save(model = None, cfg = None, save_img = True, tag='default'):
    print("=> extracting templates for PEM ...")
    tem_path = cfg.template_dir
    all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, cfg.test_dataset)
    with torch.no_grad():
        all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    print("=> loading input data for PEM ...")
    input_data, img, whole_pts, model_points, detections = get_test_data(
        cfg.rgb_path, cfg.depth_path, cfg.cam_path, cfg.cad_path, cfg.seg_path, 
        cfg.det_score_thresh, cfg.test_dataset, SCENE_ID
    )
    ninstance = input_data['pts'].size(0)
    
    torch.cuda.empty_cache()

    print("=> running PEM ...")
    with torch.no_grad():
        input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
        input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
        # del all_tem_pts, all_tem_choose, all_tem_feat # GPU memory 확보

        out = model(input_data)

    if 'pred_pose_score' in out.keys():
        pose_scores = out['pred_pose_score'] * out['score']
    else:
        pose_scores = out['score']
    pose_scores = pose_scores.detach().cpu().numpy()
    pred_rot = out['pred_R'].detach().cpu().numpy()
    pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

    # get Cam to Object Homogeneous Matrix with best score
    best_idx = pose_scores.argmax()
    R_best = pred_rot[best_idx]      # shape: (3,3) 또는 (9,) or (4,4) 일 수 있음 (아래 참고)
    t_best = pred_trans[best_idx]    # shape: (3,)
    H_OC_best = gf.to44(R_best, t_best) # Cam 시점 Object 위치 HM

    print("=> saving PEM results ...")
    os.makedirs(f"{cfg.output_dir}", exist_ok=True)
    for idx, det in enumerate(detections):
        detections[idx]['score'] = float(pose_scores[idx])
        detections[idx]['R'] = list(pred_rot[idx].tolist())
        detections[idx]['t'] = list(pred_trans[idx].tolist())

    with open(os.path.join(cfg.output_dir, f'detection_pem_{tag}_{object_id}.json'), "w") as f:
        json.dump(detections, f)

    print("=> visualizating PEM ...")
    save_path = os.path.join(f"{cfg.output_dir}", f'vis_pem_{tag}_{object_id}.png')
    valid_masks = pose_scores == pose_scores.max()
    # valid_masks = pose_scores >= pose_scores.min()
    # valid_masks = pose_scores > 0.01
    K = input_data['K'].detach().cpu().numpy()[valid_masks]

    if save_img:
        vis_img = PEM_visualize(img, pred_rot[valid_masks], pred_trans[valid_masks], model_points*1000, K, save_path)
        vis_img.save(save_path)

    torch.cuda.empty_cache()

    return H_OC_best

# ---- Grasping Alg. + EE pose calculation ----


def get_object_grasping(target_mesh_file = cad_path,
                        # top_pairs = 100,
                        H_OC = None, show = True, save = True, save_path = None):
    """
    top_pairs: pair 개수
    H_OC: Cam 기준 Object 위치 HM
    """
    result = gf.compute_best_patch_pairs(
        mesh_path=target_mesh_file,
        mesh_max_triangles = 1000,
        angle_deg=15,              # 패치 병합 허용 각도 (↑면 패치 수 ↓)
        # coplanar_tol=1,            # 공면성 허용 오차 (↑면 패치 수 ↓)
        min_opening=1.0,          # 그리퍼 최소 개구(mm)
        max_opening=140.0,         # 그리퍼 최대 개구(mm)
        angle_tolerance_deg=15,    # 패치 페어 정반대 threshold
        top_k=100                  # 상위 패치 페어 후보 개수
    )
    reports = gf.check_gripper_feasibility_faces_with_yaw(result)
    feasible_r = [r for r in reports if r.get('feasible')]
    print(f'[GRP] feasible gripping sol.: {len(feasible_r)}')

    mesh_patches = result['mesh_patches']
    pad_params = gf.PadParams() # 패드 파라미터 미리 로드

    feasible_pairs = []
    H_dict = {"EE origin": np.eye(4,4)} # EE 좌표계 원점 
    # for k in range(min(top_pairs, len(feasible_r))):
    for k in range(len(feasible_r)):
        report = reports[k]
        pi = result['patches'][report['patch_i']]
        pj = result['patches'][report['patch_j']]
        ni = gf.unit(np.asarray(pi["normal"], float))
        nj = gf.unit(np.asarray(pj["normal"], float))
        fi = report['face_i']
        fj = report['face_j']
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)
        yaw_deg = report['feasible_yaw']

        H_OG, stroke = gf.build_gripper_pose_obj_OPE({"centroid": ci, "normal": ni}, {"centroid": cj, "normal": nj}, yaw_deg, H_OC=H_OC)
        H_EdEn, H_OdEn = gf.ee_delta_pose_des(H_OC, H_OG)
        
        r_quat = Rotation.from_matrix(H_EdEn[:3,:3]).as_quat()

        # pair 검사 및 실제로 feasible한 pair 만 남김
        # 그리퍼 접근 각도 45 이하인 경우
        z_H_EdEn = H_EdEn[:3,2] # z축 각도 필터링용
        if np.dot(z_H_EdEn, np.array([0,0,1])) < 0.5: 
            continue
        # # 그리퍼 패드 - 바닥 간섭 검사
        pad_diagonal = np.sqrt(pad_params.pad_w**2 + pad_params.pad_h**2)
        pad_radius = pad_diagonal / 2.0
        H_OdEn_rot = H_OdEn[:3,:3]
        O_rotated = H_OdEn_rot @ mesh_patches.vertices.T
        z_check = O_rotated[-1,:].min() + pad_radius
        ci_world = H_OdEn_rot @ ci
        cj_world = H_OdEn_rot @ cj
        # print(f'{ci[2]:.3f}, {cj[2]:.3f}, {ci_world[2]:.3f}, {cj_world[2]:.3f}')
        if ci_world[2] < z_check or cj_world[2] < z_check:
            continue

        t_quat = H_EdEn[:3,3] / 1000
        res = {"pose_quat": np.concatenate([t_quat, r_quat]),
               "stroke": stroke}
        feasible_pairs.append(res)

        if show:
            H_dict[f"pair {k+1}"] = H_EdEn

    if show:
        gfv.visualize_frames(H_dict, scale=100, H_OdEn=H_OdEn, result=result, save=save, save_path=save_path)

    return feasible_pairs


# ---- Realtime RGBD camera stream func. (Intel Realsense) ----
def RT_get_realsense_stream():
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # scene_camera.json create
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale} (meters per unit)")
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    cam_K = [
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ]
    scene_camera_info = {}
    scene_camera_info[SCENE_ID] = {
        "cam_K": [v for row in cam_K for v in row],
        "depth_scale": depth_scale * 1000.0,  # mm 단위
        "view_level": 0
    }
    # dir_path = f'./RUN_result/test_rgbd/{}'
    dir_path = f'./RUN_result/test_rgbd/{time.strftime("%Y%m%d")}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(cam_path, "w") as f:
        json.dump(scene_camera_info, f, indent=4)

    return pipeline, align

def RT_get_one_rgbd_frame(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image


# ---- COMM func ----
def send_command(IP, PORT, res):
    # 데이터 문자열
    dx, dy, dz, dqx, dqy, dqz, dqw = res['pose_quat']
    stroke = res['stroke']
    message = f"{dx},{dy},{dz},{dqx},{dqy},{dqz},{dqw},{stroke}"
    
    print(f"Connecting to {IP}:{PORT}...")
    
    try:
        # TCP 소켓 생성 및 서버에 연결
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((IP, PORT))
            
            # 메시지를 UTF-8 바이트로 인코딩하여 전송
            s.sendall(message.encode('utf-8'))
            
            print(f"Successfully sent command: {message}")
            
    except ConnectionRefusedError:
        print(f"!!! Error: Connection refused. Check ROS PC")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# ---- Main loop  ----
def RT_inference(comm = False, IP = '', PORT = ''):
    global SCENE_ID
    global SCENE_DATE

    # 1. 모델 미리 로드 (권장: 느려도 최초 1회만!)
    model_ism = ISM_load()
    cfg = PEM_init()
    model_pem = PEM_load(cfg)
    
    # 2. 카메라 오픈
    pipeline, align = RT_get_realsense_stream()

    print("Realtime framing... press 'i' to inference / 'ESC' to exit.")
    while True:
        rgb_frame, depth_frame = RT_get_one_rgbd_frame(pipeline, align)
        if rgb_frame is None:
            continue

        # 실시간 디스플레이
        cv2.imshow("RGB Stream", rgb_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # 'i' 입력시 ISM+PEM 수행
        if key == ord('i'):
            # 1. 프레임 저장 
            now_str = time.strftime("%Y%m%d_%H%M%S")
            rgb_save_path =   f"./RUN_result/input/now_rgb_{now_str}.png"
            depth_save_path = f"./RUN_result/input/now_depth_{now_str}.png"
            cv2.imwrite(rgb_save_path, rgb_frame)
            cv2.imwrite(depth_save_path, depth_frame)

            # 2. ISM 추론
            print("[Realtime] ISM running...")
            ism_result = ISM_run_save(
                ISMmodel = model_ism, 
                rgb_path = rgb_save_path, 
                depth_path = depth_save_path,
                save_img = True, 
                tag = now_str
            )
            print("[Realtime] ISM done!")

            # 3. PEM 추론
            print("[Realtime] PEM running...")
            # config에서 해당 프레임 경로로 세팅 (필요시)
            cfg.rgb_path = rgb_save_path
            cfg.depth_path = depth_save_path
            cfg.seg_path = f"{output_dir}/detection_ism_{now_str}_{object_id}.json"  # ISM 저장 경로
            H_OC_best = PEM_run_save(model = model_pem,
                                     cfg = cfg,
                                     save_img = True,
                                     tag = now_str) # best HM
            print("[Realtime] PEM done!")

            # 4. 그리핑 결과 전송 
            gsolpath = f"{output_dir}/vis_grasping_sol_{now_str}_{object_id}.html"
            grip_result = get_object_grasping(H_OC=H_OC_best, save_path = gsolpath)
            # for grip_res in grip_result:
            #     print(grip_res)
            grip_res = grip_result[0] # 첫번째 후보
            if comm == True:
                send_command(IP, PORT, grip_res)

            # +. 결과 이미지 표시 
            # ism_vis_path = f"{output_dir}/vis_ism_{now_str}_{object_id}.png"
            # if os.path.exists(ism_vis_path):
            #     ism_vis_img = cv2.imread(ism_vis_path)
            #     cv2.imshow("ISM Result", ism_vis_img)
            #     cv2.waitKey(1)
            pem_vis_path = f"{output_dir}/vis_pem_{now_str}_{object_id}.png"
            if os.path.exists(pem_vis_path):
                pem_vis_img = cv2.imread(pem_vis_path)
                cv2.imshow("PEM Result", pem_vis_img)
                cv2.waitKey(1)

        elif key == 27:  # ESC
            break

    pipeline.stop()
    cv2.destroyAllWindows()

################################# Sequence (Realtime) #################################
if __name__ == "__main__":
    RT_inference(comm=True, IP='192.168.0.22', PORT=9900)