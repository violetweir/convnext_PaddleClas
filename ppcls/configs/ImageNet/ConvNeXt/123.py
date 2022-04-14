import os

if __name__=="__main__":
    
    print(os.getcwd())
    print('解压文件夹')
    os.system('git clone https://github.com/violetweir/ConvNeXt_PaddleClas2.git')
    print('解压完毕')
    print('安装包')
    os.system("cd ConvNeXt_PaddleClas2 && pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple")
    os.system("pip install pycocotools -i https://mirror.baidu.com/pypi/simple")
    os.system('tar -xvf ')
    print('开始训练')
    os.system("cd ConvNeXt_PaddleClas2 && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval --use_vdl=true --vdl_log_dir=vdl_dir/scalar")
    # os.system("mv PaddleDetection/output/ppyolov2_r50vd_dcn_365e_coco/best_model.pdparams /root/paddlejob/workspace/output")
    # os.system("mv ./output/ppyolov2_r50vd_dcn_365e_coco/best_model.pdparams /root/paddlejob/workspace/output")
    # os.system("mv ./output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams /root/paddlejob/workspace/output")
    # os.system("mv PaddleDetection/output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams /root/paddlejob/workspace/output")
    # os.system("mv ./vdl_dir/scalar  /root/paddlejob/workspace/output")
    # os.system("mv PaddleDetection/vdl_dir/scalar  /root/paddlejob/workspace/output")