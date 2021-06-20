#import cv2
import os, datetime, json, subprocess
import ffmpeg, srt
from google.api_core import operation
from google.cloud.speech_v1.services.speech import client
from google.cloud import speech
#from google.protobuf.json_format import MessageToJson

class VideoConverter:
    def __init__(self, src='', output_dir=''):
        self.setSrc(src)
        self.des_dir = output_dir if output_dir else os.path.dirname(src)
        self.supported_video_output_format = dict([('mp4', 'h264')])
        self.video_temp = []
        self.clip_temp = []
        self.audio_temp = []
        self.srt_temp = []
        self.sst_temp = []  # save speech recognition result to avoid extra cost
        self.video_list_file = 'video_list.txt'

    def setSrc(self, src):
        self.src = src
        self.src_basename = os.path.basename(src) if src else ''

    # TODO: Abandon ffmpeg model, use subprocess call ffmpeg command instead
    def convertTo(self, src='', video_type='mp4', bit_rate=0):
        if not src: src = self.src

        if not video_type in self.supported_video_output_format:
            print('输出文件格式不支持, 支持文件为: ' + ', '.join(self.supported_video_output_format))
            return 0

        des = src.rsplit('.', 1)[0] + '.' + video_type
        video = ffmpeg.input(src)
        if bit_rate > 0:
            video.output(des, video_bitrate=bit_rate).run()
            return 1
        else:
            video.output(des).run()
            return 1

    def extract_audio(self, input='', output_dir='', extension='wav', channel=1, sample_rate=16000):
        if not input: input = self.src
        output = output_dir + '/' + os.path.basename(input) + '.' + extension
        if not os.path.isfile(output):
            print(f'正在将{os.path.basename(input)}转换为音频格式...')
            self.audio_temp.append(output)
            subprocess.run(['ffmpeg','-i',input,'-v','error','-ac',str(channel),'-ar',str(sample_rate),'-vn',output])
        else:
            print(f'文件{os.path.basename(output)}已经存在')

    """
    " OpenCV Region
    def changeBackground(self):
        replace_bg = cv2.imread("J03751_1200x.jpg")
        cap=cv2.VideoCapture(self.src)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        replace_bg = cv2.resize(replace_bg, (w, h), interpolation=cv2.INTER_AREA)
        j=0
        #bgSub = cv2.createBackgroundSubtractorKNN()
        while True:
            ret,frame=cap.read()
            if not ret: break
            if(j==0):
                bg=frame.copy().astype("float")
            if(j<45):
                cv2.accumulateWeighted(frame,bg,0.5)
                j=j+1
            diff=cv2.absdiff(frame,bg.astype("uint8"))
            diff=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
            _,diff=cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
            
            fgImg = cv2.bitwise_and(frame, frame, mask=diff)
            dst = cv2.add(replace_bg, fgImg)
            cv2.imshow('F', dst)
            if(cv2.waitKey(1) & 0XFF==ord('q')):
                break
        cap.release()
        cv2.destroyAllWindows()

    def remove_water_mark(self, video_type='mp4'):
        cap = cv2.VideoCapture(self.src)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*self.supported_video_output_format[video_type])
        writer = cv2.VideoWriter(self.des, fourcc, fps, (w, h))
        mask = cv2.imread('mask.png', 0)
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            removed = cv2.inpaint(frame, resized_mask, 3, cv2.INPAINT_TELEA)
            writer.write(removed)

        cap.release()
        writer.release()
    """
    
    def split_video(self, input='', output_folder='', video_length=0, sec_limit=0):
        """ 分割视频
        "   return 分割后的视频片段路径
        "   input: 输入文件路径,默认为该类的src
        "   output_folder: 分割后视频存放目录，默认为/TEMP
        "   sec_limit: 每个视频最大时长
        """

        if not input: input = self.src
        if not output_folder: output_folder = os.path.dirname(self.src) + '/TEMP'
        if sec_limit == 0: return [input]

        res = []
        base_command = ['ffmpeg','-i', input, '-v', 'quiet']

        # get video length if not provided
        if video_length == 0:
            duration = subprocess.run(f'ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input}"', 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            video_length = float(duration.stdout)

        # make sure output folder is ready
        if not os.path.isdir(output_folder): os.makedirs(output_folder)
        
        if video_length < sec_limit: return [input]

        start_sec = 0
        for i in range(int(video_length / sec_limit) + 1):
            if start_sec == video_length: break

            end_sec = (start_sec + sec_limit) if start_sec + sec_limit < video_length else video_length
            src_path_arr = self.src_basename.rsplit('.', 1)
            file_path = output_folder + '/' + src_path_arr[0] + '_' + str(i+1) + '.' + src_path_arr[1]
            res.append(file_path)

            if not os.path.isfile(file_path):
                print(f'正在将视频剪辑为从{start_sec}秒到{end_sec}秒的片段')
                out_command = base_command + ['-ss', str(start_sec), '-to', str(end_sec), '-c', 'copy', file_path]
                subprocess.run(out_command)

            start_sec = end_sec

        self.video_temp = res
        return res

    def __local_transcript(self, videopath='', subtitle_duration=3):
        # if already has subtitle file skip speech recognition
        subtitle_path = videopath.rsplit('.', 1)[0] + '.srt'

        if not os.path.isfile(subtitle_path):
            # prepare speech recognition google api
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code = 'zh-CN',
                sample_rate_hertz = 16000,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True
            )

            # convert to audio format
            audio_path = videopath.rsplit('.')[0] + '.wav'
            if not os.path.isfile(audio_path):
                print('正在将视频片段转换为音频格式...')
                self.audio_temp.append(audio_path)
                subprocess.run(['ffmpeg','-i',videopath,'-v','quiet','-ac','1','-ar','16000','-vn',audio_path])

            with open(audio_path, "rb") as f:
                content = f.read()
            audio = speech.RecognitionAudio(content=content)
            print('正在使用谷歌API进行语音识别中...')
            response = client.recognize(request={"config": config, "audio": audio})
            self.generate_subtitle(response.results, subtitle_duration, fileName=subtitle_path)
        
        outputPath = videopath.rsplit('.', 1)[0] + '_sub.mp4'
        if not os.path.isfile(outputPath):
            self.combineSubtitles(subtitle_path, videopath, outputPath)
        else:
            print('嵌有字幕的视频片段已存在，跳过')

    def __cloud_transcript(self, audioURI='', subtitle_duration=3):
        # prepare output file container
        subtitle_path = audioURI.split('/', 3)[-1] + '.srt'
        if not os.path.isdir(os.path.dirname(subtitle_path)):
            os.makedirs(os.path.dirname(subtitle_path))

        if not os.path.isfile(subtitle_path):
            # no subtitle file, request speech recognition api
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code = 'zh-CN',
                sample_rate_hertz = 16000,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True
            )

            audio = speech.RecognitionAudio(uri=audioURI)
            response = client.long_running_recognize(config=config, audio=audio)


    def transcript_video(self, link='', split=False, subtitle_duration=3):
        if not link: link = self.src

        # Check video length
        duration = subprocess.run(f'ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input}"', 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        video_length = float(duration.stdout)

        if video_length < 60:
            # use local transcription method directly
            self.__local_transcript(link, subtitle_duration)
        elif video_length >= 60 and split:
            # split into partial videos with duration less than 1 minute, and than use local transcription method
            split_v = self.split_video(link, video_length=video_length, sec_limit=58)
            for v in split_v:
                self.__local_transcript(v, subtitle_duration)
        else:
            # use cloud transcription method
            self.__cloud_transcript(link, subtitle_duration)

    def generate_subtitle(self, text_result, bin_size=3, fileName='sub.srt'):
        """ 生成字幕文件
        "   return None
        "   text_result: 语音识别后的的结果
        "   bin_size: 对应subtitle_duration, 规定每行字幕最大显示时长，默认3秒
        "   fileName: 输出字幕文件路径
        """

        transcriptions = []
        index = 0
    
        for result in text_result:
            try:
                if result.alternatives[0].words[0].start_time.seconds:
                    # bin start -> for first word of result
                    start_sec = result.alternatives[0].words[0].start_time.seconds 
                    start_microsec = result.alternatives[0].words[0].start_time.microseconds * 0.001
                else:
                    # bin start -> For First word of response
                    start_sec = 0
                    start_microsec = 0 
                end_sec = start_sec + bin_size # bin end sec
                
                # for last word of result
                last_word_end_sec = result.alternatives[0].words[-1].end_time.seconds
                last_word_end_microsec = result.alternatives[0].words[-1].end_time.microseconds
                
                # bin transcript
                transcript = result.alternatives[0].words[0].word
                
                index += 1 # subtitle index

                for i in range(len(result.alternatives[0].words) - 1):
                    try:
                        word = result.alternatives[0].words[i + 1].word
                        word_start_sec = result.alternatives[0].words[i + 1].start_time.seconds
                        word_start_microsec = result.alternatives[0].words[i + 1].start_time.microseconds
                        word_end_sec = result.alternatives[0].words[i + 1].end_time.seconds

                        if word_end_sec < end_sec:
                            transcript += word
                        else:
                            previous_word_end_sec = result.alternatives[0].words[i].end_time.seconds
                            previous_word_end_microsec = result.alternatives[0].words[i].end_time.microseconds
                            
                            # append bin transcript
                            line = srt.Subtitle(index, datetime.timedelta(seconds=start_sec, microseconds=start_microsec), datetime.timedelta(seconds=previous_word_end_sec, microseconds=previous_word_end_microsec), transcript)
                            transcriptions.append(line)
                            
                            # reset bin parameters
                            start_sec = word_start_sec
                            start_microsec = word_start_microsec
                            end_sec = start_sec + bin_size
                            transcript = result.alternatives[0].words[i + 1].word
                            
                            index += 1
                    except IndexError as ie:
                        print('生成字幕时发生错误: ' + ie)

                # append transcript of last transcript in bin
                transcriptions.append(srt.Subtitle(index, datetime.timedelta(seconds=start_sec, microseconds=start_microsec), datetime.timedelta(seconds=last_word_end_sec, microseconds=last_word_end_microsec), transcript))
                index += 1
            except IndexError as ie:
                print('准备生成字幕对象时出错:' + ie)
        
        # turn transcription list into subtitles
        subtitles = srt.compose(transcriptions)
        with open(fileName, 'w', encoding='utf8') as f:
            f.writelines(subtitles)
        print('已生成字幕文件, 文件路径: ' + fileName)
        self.srt_temp.append(fileName)

    def combineSubtitles(self, subtitle_path='', video_path='', output_path=''):
        if not video_path:
            video_path = self.src
            subtitle_path = self.src.rsplit('.', 1)[0] + '.srt'

        if not output_path: output_path = video_path.rsplit('.', 1)[0] + '_sub.mp4'
        self.clip_temp.append(output_path)

        if not os.path.isfile(output_path):
            print('正在嵌入字幕到视频片段中...')
            subprocess.call(['ffmpeg', '-i', video_path, '-v', 'error', '-vf', 'subtitles='+subtitle_path, output_path])
        else:
            print('带字幕视频文件已存在, 跳过')

        with open(self.video_list_file, 'a', encoding='utf8') as v_list:
            v_list.write(f"file '{output_path}'\n")

    def combineVideos(self, video_list_fp=''):
        if not video_list_fp: video_list_fp = self.video_list_file
        if os.path.isfile(video_list_fp):
            fn_temp = self.src.rsplit('.', 1)
            output_fp = fn_temp[0] + '_subbed.' + fn_temp[1]
            print('正在将视频片段合成...')
            subprocess.run(['ffmpeg','-f','concat','-i',video_list_fp,'-v','error','-codec','copy',output_fp])
        else:
            print('缺少需合成的文件列表')

    def clear_temp_except(self, *temp_to_keep):
        print('正在清理临时文件...')

        if 'video' not in temp_to_keep:
            for v in self.video_temp:
                if os.path.isfile(v): os.remove(v)
            self.video_temp = []
        if 'audio' not in temp_to_keep:
            for a in self.audio_temp:
                if os.path.isfile(a): os.remove(a)
            self.audio_temp = []
        if 'subtitle' not in temp_to_keep:
            for s in self.srt_temp:
                if os.path.isfile(s): os.remove(s)
            self.srt_temp = []
        if 'clip' not in temp_to_keep:
            for c in self.clip_temp:
                if os.path.isfile(c): os.remove(c)
            self.clip_temp = []
        if 'list' not in temp_to_keep:
            if os.path.isfile(self.video_list_file): os.remove(self.video_list_file)

def main():
    #vc = VideoConverter('gs://lele-vid-stt-storage/高中数学/高中必修1/二分法/21137.mp4')
    #vc.cloud_transcript()
    #file_list = vc.split_video(sec_limit=58)
    #for f in file_list:
    #    vc.transcript_video(f, 4)
    #vc.combineVideos()
    #vc.clear_temp_except('subtitle', 'clip')
    count = 0
    vc = VideoConverter()
    for root,_,files in os.walk('Test/高中数学'):
        for f in files:
            #if count >= 1: return
            file_path = (root + '/' + f).replace('\\', '/')
            #print(file_path)
            #print(file_path.replace('\\', '/'))
            vc.setSrc(file_path)
            #splitted_fp = vc.split_video(sec_limit=58)
            #for s_fp in splitted_fp:
            #    vc.transcript_video(s_fp, 4)
            #vc.combineVideos()
            #vc.clear_temp_except('subtitle', 'clip')
            out_dir = 'Out/' + os.path.dirname(file_path)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            vc.extract_audio(output_dir=out_dir)
            count += 1
    print(count)

if __name__ == '__main__':
    main()