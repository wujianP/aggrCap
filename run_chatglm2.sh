cd /discobox/wjpeng/code/202312/aggrCap
conda activate /discobox/wjpeng/env/clip/
JOB_ID=0
export CUDA_VISIBLE_DEVICES=$((JOB_ID % 8))

python main.py \
--job-num=8 \
--job-id=$JOB_ID \
--batch-size=8 \
--caption-path='/DDN_ROOT/jia/code/Open-VCLIP-V2/video_description_gen/flan_xxl_record_uniform.pth' \
--model='THUDM/chatglm2-6b' \
--output='outputs/chatglm2-6b' \
--save-freq=10 \
--print-freq=1

