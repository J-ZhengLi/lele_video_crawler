#import cv2
import os, datetime, subprocess, threading, time
import srt
from queue import Queue
from google.cloud.speech_v1.services.speech import client
from google.cloud import speech

class VideoFactory:
    def __init__(self, src='', output_dir=''):
        self.setSrc(src)
        self.des_dir = output_dir if output_dir else os.path.dirname(src)
        self.supported_video_output_format = dict([('mp4', 'h264')])
        self.supported_audio_format = ['wav', 'mp3', 'ogg']
        self.video_list_file = 'video_list.txt'

        # store file cache
        self.video_temp = []
        self.clip_temp = []
        self.audio_temp = []
        self.srt_temp = []

        self.splitted = False

    def setSrc(self, src):
        """
        设置输入文件（夹）路径
        """
        self.src = src
        self.src_basename = os.path.basename(src) if src else ''

    def convertTo(self, src='', extension='mp4', bit_rate=0):
        """
        视频格式转换函数
        --------------
        args
        src : str
            输入路径, 默认与此类的src相同
        extension : str
            要转换的文件格式, 默认mp4
        bit_rate : int
            输出格式的视频比特率, 默认为0(即不更改)
        ---------------------------------------
        return
        int
            转换成功的个数
        """

        if not src: src = self.src
        if not extension in self.supported_video_output_format:
            print('输出文件格式不支持, 支持文件为: ' + ', '.join(self.supported_video_output_format))
            return 0

        command = ['ffmpeg','-i',src,'-v','error','-codec:v',self.supported_video_output_format[extension]]
        des = src.rsplit('.', 1)[0] + '.' + extension
        if bit_rate > 0:
            command += ['-b:v',str(bit_rate),des]
        else:
            command += [des]
        print(f'\"{src}\"文件转换中...')
        subprocess.run(command)
        return 1

    def extract_audio(self, src='', output_dir='', extension='wav', channel=1, sample_rate=16000):
        """
        视频文件转音频 (提取音频)
        --------------
        args
        src : str
            输入路径, 默认与此类的src相同
        output_dir : str
            输出文件夹路径, 若留空则存放在当前目录下
        extension : str
            输出音频文件的格式, 默认为wav
        channel : int
            输出音频文件声道, 默认为1（单声道）
        sample_rate : int
            输出音频文件的采样率, 默认为16k
        """
        if not src: src = self.src
        if output_dir and not (output_dir[-1] == '/' or output_dir[-1] == '\\'): output_dir += '/'
        if not extension in self.supported_audio_format:
            print(f'暂不支持输出格式为{extension}的文件.')
            return
        if channel < 1 or channel > 2:
            print('音频声道不支持.')
            return

        temp_path = os.path.relpath(src).rsplit('\\', 1)
        output_dir += temp_path[0]
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        output = output_dir + '/' + temp_path[1] + '.' + extension
        if os.path.isfile(output):
            print(f'路径下的文件({output})已经存在, 自动跳过')
            return

        command = ['ffmpeg','-i',src,'-v','error','-ac',str(channel)]
        if sample_rate > 0:
            command += ['-ar',str(sample_rate),'-vn',output]
        else:
            command += ['-vn',output]

        print(f'正在从{src}提取音频...')
        #print(output)
        subprocess.run(command)
        self.audio_temp.append(output)

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
    
    def split_video(self, src='', output_dir='', video_length=0, sec_limit=0):
        """
        视频分段函数-按时长将视频分为片段
        --------------
        args
        src : str
            输入路径, 默认与此类的src相同
        output_dir : str
            输出文件夹路径, 若留空则存放在当前目录下
        video_length : int
            视频总时长, 若为0则实时获取,请确保ffprobe已安装, 默认为0
        sec_limit : int
            需要分割的每个片段最大时长(以秒为单位), 默认为0(即不做处理)
        --------------
        return
        res : list<str>
            此文件所有分割后片段的路径
        """

        if not src: src = self.src
        if not output_dir: output_dir = os.path.dirname(self.src) + '/TEMP'
        if sec_limit == 0: return [src]

        res = []

        # get video length if not provided
        if video_length == 0:
            duration = subprocess.run(f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{src}"', 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            video_length = float(duration.stdout)
        if video_length < sec_limit: return [src]

        # make sure output folder is ready
        if not os.path.isdir(output_dir): os.makedirs(output_dir)

        start_sec = 0
        for i in range(int(video_length / sec_limit) + 1):
            if start_sec == video_length: break

            end_sec = (start_sec + sec_limit) if start_sec + sec_limit < video_length else video_length
            src_path_arr = os.path.basename(src).rsplit('.', 1)
            file_path = output_dir + '/' + src_path_arr[0] + '_' + str(i+1) + '.' + src_path_arr[1]
            res.append(file_path)

            if not os.path.isfile(file_path):
                print(f'正在将{start_sec}秒到{end_sec}秒的视频片段存放到路径{file_path}下')
                subprocess.run(f'ffmpeg -i {src} -v error -ss {str(start_sec)} -to {str(end_sec)} -c copy {file_path}')
            start_sec = end_sec

        self.video_temp = res
        self.splitted = True
        return res

    def __local_transcript(self, videopath='', subtitle_duration=3):
        # if already has subtitle file skip speech recognition
        subtitle_path = videopath.rsplit('.', 1)[0] + '.srt'
        self.srt_temp.append(subtitle_path)

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

        return subtitle_path

    def __cloud_transcript(self, audioURI='', subtitle_duration=3):
        # prepare output file container
        subtitle_path = audioURI.split('/', 3)[-1].rsplit('.', 1)[0]  + '.srt'
        self.srt_temp.append(subtitle_path)

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
            operation = client.long_running_recognize(config=config, audio=audio)
            for retry_count in range(3):
                response = operation.result()
                if response:
                    break
                if retry_count > 0:
                    print(f'正在进行第{retry_count}次重试中...')

            self.generate_subtitle(response.results, subtitle_duration, subtitle_path)
        else:
            print('字幕文件已存在, 无需调用API转录')
        
        return subtitle_path
        #videopath = 'Test/' + subtitle_path.rsplit('.', 1)[0]
        #outputPath = subtitle_path.rsplit('.', 2)[0] + '_sub.mp4'
        #self.combineSubtitles(subtitle_path, videopath, outputPath)

    def transcript_video(self, src='', output_dir='', isGCS=False, subtitle_duration=3):
        """
        视频转录函数
        --------------
        args
        src : str
            输入路径, 默认与此类的src相同
        output_dir : str
            输出文件夹路径, 若留空则存放在当前目录下
        isGCS : bool
            该文件路径是否为Google Cloud Storage(谷歌云)路径, 如果为本地路径请保持默认值
        subtitle_duration : int
            转录后每行字幕显示的时间秒数, 默认3秒
        -------------------------------------
        return
        res : list<tuple(str, str)>
            原视频路径以及字幕文件路径, 便于后期合并
        """

        if not src: src = self.src
        #if not output_dir: output_dir = os.path.dirname(self.src) + '/output'
        #if not os.path.isdir(output_dir): os.makedirs(output_dir)

        if isGCS:
            return [(src, self.__cloud_transcript(src, subtitle_duration))]
        else:
            # split video to make sure it is less than 1 minute
            # use 58 seconds instead of 60 to make sure
            res = []
            splitted_paths = self.split_video(src, sec_limit=58)
            for sp in splitted_paths:
                sub_path = self.__local_transcript(sp, subtitle_duration)
                res.append((sp, sub_path))
            return res

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

    def combineSubtitles(self, subtitle_path, video_path='', output_dir=''):
        """
        将字幕嵌入视频
        --------------
        args
        subtitle_path : str
            输入的字幕文件路径, 不得为空
        video_path : str
            输入的视频文件路径, 若留空则默认为此类的src
        output_dir : str
            输出文件夹路径, 默认和输入视频文件目录相同
        """
        if not video_path: video_path = self.src
        if not output_dir: output_dir = os.path.dirname(video_path)
        if self.splitted: self.clip_temp.append(output_dir)

        output_path = output_dir + '/' + os.path.basename(video_path).rsplit('.', 1)[0] + '_sub.mp4'
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

    def clear_temp(self, clear_all=False, *temp_to_clear):
        """
        临时文件清理功能
        --------------
        args
        clear_all : bool
            是否清除所有缓存, 可以简化输入步骤
        temp_to_keep : *str
            指定要清除的文件类型, 只有clear_all为False时生效\\
            文件类型: 
                video(分割后视频片段)\\
                audio(用作语音识别的音频文件)\\
                subtitle(转录后的字幕文件)\\
                clip(视频合并前带字幕的视频片段)\\
                text(用作视频合并的片段路径文本)
        """

        if clear_all or 'video' in temp_to_clear:
            for v in self.video_temp:
                if os.path.isfile(v): os.remove(v)
            self.video_temp = []
        if clear_all or 'audio' in temp_to_clear:
            for a in self.audio_temp:
                if os.path.isfile(a): os.remove(a)
            self.audio_temp = []
        if clear_all or 'subtitle' in temp_to_clear:
            for s in self.srt_temp:
                if os.path.isfile(s): os.remove(s)
            self.srt_temp = []
        if clear_all or 'clip' in temp_to_clear:
            for c in self.clip_temp:
                if os.path.isfile(c): os.remove(c)
            self.clip_temp = []
        if clear_all or 'text' in temp_to_clear:
            if os.path.isfile(self.video_list_file): os.remove(self.video_list_file)

# ------------------------- END OF CLASS ---------------------------

lock = threading.Lock()
vc = VideoFactory()
def thread_task(filepath):
    #lock.acquire()
    vc.transcript_video(filepath, isGCS=True, subtitle_duration=4)
    #vc.clear_temp('text')
    #lock.release()

def get_videopaths(folder_path):
    count = 0
    if not os.path.isdir(folder_path): return count

    with open('transcript_list.txt', 'w', encoding='utf8') as f:
        for root, _, files in os.walk(folder_path):
            for file in files:
                filepath = root.replace('\\', '/') + '/' + file
                f.write(filepath + '\n')
                count += 1
    return count

def multithread_transcript(input_queue, thread_count=20, timeout_sec=10):
    threads = []
    if input_queue.empty() or thread_count < 1: return
    while not input_queue.empty():
        if threading.active_count() < thread_count:
            t = threading.Thread(target=thread_task, args=[input_queue.get()])
            t.start()
            threads.append(t)
        else:
            print('等待中...')
            time.sleep(timeout_sec)

    for thread in threads:
        thread.join()

# set GOOGLE_APPLICATION_CREDENTIALS=F:/STT.json

def main():
    #res = get_videopaths('Test/初中数学')
    #fileQueue = Queue()
    with open('transcript_list.txt', 'r', encoding='utf8') as file:
        for line in file.readlines():
            video_path = line.strip()
            file_path = 'gs://lele-vid-stt-storage/' + video_path.split('/', 1)[1] + '.wav'

            #fileQueue.put(file_path)
            (_, srt_file) = vc.transcript_video(file_path, True, 4)[0]
            output_path = srt_file.rsplit('/', 1)[0]
            #print(output_path)
            vc.combineSubtitles(srt_file, video_path, output_path)
            #vc.clear_temp_except('subtitle', 'clip')

    #multithread_transcript(fileQueue)

if __name__ == '__main__':
    main()