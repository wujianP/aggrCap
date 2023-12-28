JOB_ID=0
cd /discobox/wjpeng/code/202312/aggrCap
conda activate /discobox/wjpeng/env/clip/
export CUDA_VISIBLE_DEVICES=$((JOB_ID % 8))

python main.py \
--job-num=8 \
--job-id=$JOB_ID \
--batch-size=8 \
--caption-path='/DDN_ROOT/jia/code/Open-VCLIP-V2/video_description_gen/flan_xxl_record_uniform.pth' \
--model='Qwen/Qwen-7B-Chat' \
--output='outputs/qwen-7b-chat' \
--save-freq=10 \
--print-freq=5
