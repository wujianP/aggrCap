cd /discobox/wjpeng/code/202312/aggrCap
conda activate /discobox/wjpeng/env/clip/

python main.py \
--batch-size=8 \
--caption-path='/DDN_ROOT/jia/code/Open-VCLIP-V2/video_description_gen/flan_xxl_record_uniform.pth' \
--model='lmsys/vicuna-7b-v1.5' \
--output='outputs' \
--save-freq=10


# Fastchat
cd /discobox/wjpeng/code/202312/aggrCap
conda activate /discobox/wjpeng/env/clip/

python main.py \
--batch-size=8 \
--caption-path='/DDN_ROOT/jia/code/Open-VCLIP-V2/video_description_gen/flan_xxl_record_uniform.pth' \
--model='lmsys/fastchat-t5-3b-v1.0' \
--output='outputs' \
--save-freq=10

# QWen
cd /discobox/wjpeng/code/202312/aggrCap
conda activate /discobox/wjpeng/env/clip/

python main.py \
--batch-size=8 \
--caption-path='/DDN_ROOT/jia/code/Open-VCLIP-V2/video_description_gen/flan_xxl_record_uniform.pth' \
--model='THUDM/chatglm2-6b' \
--output='outputs' \
--save-freq=10

export CUDA_VISIBLE_DEVICES=1
cd /discobox/wjpeng/code/202312/aggrCap
conda activate /discobox/wjpeng/env/clip/

python main.py \
--batch-size=8 \
--caption-path='/DDN_ROOT/jia/code/Open-VCLIP-V2/video_description_gen/flan_xxl_record_uniform.pth' \
--model='BAAI/AquilaChat2-7B' \
--output='outputs' \
--save-freq=10




export CUDA_VISIBLE_DEVICES=1
cd /discobox/wjpeng/code/202312/aggrCap
conda activate /discobox/wjpeng/env/clip/

python qwen.py \
--caption-path='/DDN_ROOT/jia/code/Open-VCLIP-V2/video_description_gen/flan_xxl_record_uniform.pth' \
--model='BAAI/AquilaChat2-7B' \
--output='outputs' \
--save-freq=10