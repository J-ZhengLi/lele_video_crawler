from video_factory import VideoFactory

vf = VideoFactory()
vf.setSrc('Test/初中数学/初二下学期/特殊的平行四边形（三）/18717.mp4')
#vf.convertTo(extension='mp4')
vf.extract_audio()
#vf.split_video(output_dir='Test/cache', sec_limit=58)