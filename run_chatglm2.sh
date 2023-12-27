export CUDA_VISIBLE_DEVICES=1
cd /discobox/wjpeng/code/202312/aggrCap
conda activate /discobox/wjpeng/env/clip/

python main.py \
--batch-size=8 \
--caption-path='/DDN_ROOT/jia/code/Open-VCLIP-V2/video_description_gen/flan_xxl_record_uniform.pth' \
--model='THUDM/chatglm2-6b' \
--output='outputs/chatglm2-6b' \
--save-freq=10
