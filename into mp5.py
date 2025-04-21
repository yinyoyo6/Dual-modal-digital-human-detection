import os
from moviepy.editor import VideoFileClip

def video_to_audio(video_path, output_folder):
    try:
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # 获取视频文件名（不包含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 设置输出音频文件路径
        audio_path = os.path.join(output_folder, f"{video_name}.mp3")
        
        # 加载视频文件
        video = VideoFileClip(video_path)
        
        # 提取音频并保存
        video.audio.write_audiofile(audio_path)
        
        # 关闭视频文件
        video.close()
        
        print(f"音频已保存到: {audio_path}")
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        return False
  
def main():
    # 设置输入视频路径和输出文件夹
    video_path = "A:/真识/test.mp4/header_video_instant_avatar2.mp4"
    output_folder = r"a:\真识\mp5"
    
    print(f"正在处理视频: {video_path}")
    video_to_audio(video_path, output_folder)

if __name__ == "__main__":
    main()