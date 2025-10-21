import math

# 将心音信号分段，每段包含固定数量的心跳周期（默认5个）
def segment_sound(record, label, n_cycles=5, samplerate=4000):
    segement_duration = n_cycles*0.8    # 每个心跳周期近似为0.8秒（经验值）
    record_duration = len(record)/samplerate #the dataset is sampled at 4K
    segments_num = math.floor(record_duration/segement_duration)
    segment_pts = math.floor(segement_duration*samplerate)
    segs_arr = []
    single_seg = []
    for i in range(segments_num):
        single_seg = record[i*segment_pts : segment_pts*(i+1)]
        segs_arr.append([single_seg,label])
    return segs_arr
    
# 对数据集中的每条记录进行分段处理，构建统一的segments列表
def build_segements(data_arr, sr=4000):
    segments = []
    for record in data_arr:
        segments.extend(segment_sound(record[0], record[1], samplerate=sr))
    return segments