from pydub import AudioSegment
import os

def process_audio(file_path, output_dir, target_length):
    # 读取音频文件
    audio = AudioSegment.from_file(file_path)
    
    # 处理前秒部分（自动填充静音）
    first_part_duration = 4000  # 转换为毫秒
    
    frame_rate = audio.frame_rate
    channels = audio.channels

    # 去掉前秒片段
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # first_part.export(os.path.join(output_dir, f"{base_name}_part0.wav"), format="wav")
    
    # 处理剩余音频
    remaining = audio[first_part_duration:]
    target_ms = target_length * 1000
    
    # 计算需要分割的段数
    full_segments = len(remaining) // target_ms
    remainder = len(remaining) % target_ms

    # 分割完整片段
    for i in range(full_segments):
        start = i * target_ms
        segment = remaining[start:start+target_ms]
        segment.export(os.path.join(output_dir, f"{base_name}_part{i+1}.wav"), format="wav")
    
    # 处理最后不足长度的片段
    if remainder > 0:
        last_segment = remaining[-remainder:]
        silence = AudioSegment.silent(
            duration=target_ms - remainder,
            frame_rate=frame_rate
        ).set_channels(channels)
        padded = last_segment + silence
        padded.export(os.path.join(output_dir, f"{base_name}_part{full_segments+1}.wav"), format="wav")

def process_folder(input_dir, output_dir, target_length):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历处理所有音频文件
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            try:
                process_audio(filepath, output_dir, target_length)
            except Exception as e:
                print(f"处理文件 {filename} 时出错：{str(e)}")



if __name__ == "__main__":
    dir=r'D:\文件\深度学习\mzyq_audio\4.民族乐器音色打分数据库\乐器音频'
    dirnames=os.listdir(dir)
    for input_dir in dirnames:
        input_dir=os.path.join(dir,input_dir)
        output_dir=input_dir.replace('乐器音频','分割音频')
        process_folder(input_dir, output_dir, target_length=14)