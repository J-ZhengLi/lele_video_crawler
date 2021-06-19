#import cv2
import os, datetime
import ffmpeg, srt, subprocess
from google.cloud import speech

class VideoConverter:
    def __init__(self, src='', output_folder='Output'):
        self.src = src
        self.src_basename = os.path.basename(src) if src else ''
        self.des_folder = output_folder
        self.supported_video_output_format = dict([('mp4', 'h264')])
        self.video_temp = []
        self.clip_temp = []
        self.audio_temp = []
        self.srt_temp = []
        self.video_list_file = 'video_list.txt'

    def setSrc(self, src):
        self.src = src

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

    # return a list of splitted video_paths
    def split_video(self, input='', output_folder='TEMP', sec_limit=0):
        if not input: input = self.src
        if sec_limit == 0: return [input]

        res = []
        base_command = ['ffmpeg','-i', input, '-v', 'quiet']
        self.video_list_file = output_folder + '/' + self.video_list_file

        # get video length
        duration = subprocess.run(f'ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input}"', 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        duration = float(duration.stdout)

        # make sure output folder is ready
        if not os.path.isdir(output_folder): os.makedirs(os.getcwd() + '/' + output_folder)
        
        if duration < sec_limit: return [input]

        start_sec = 0
        for i in range(int(duration / sec_limit) + 1):
            if start_sec == duration: break

            end_sec = (start_sec + sec_limit) if start_sec + sec_limit < duration else duration
            src_path_arr = self.src_basename.rsplit('.', 1)
            file_path = output_folder + '/' + src_path_arr[0] + '_' + str(i+1) + '.' + src_path_arr[1]
            res.append(file_path)
            if not os.path.isfile(file_path):
                print(f'正在将视频剪辑为从{start_sec}秒到{end_sec}秒的片段')
                out_command = base_command + ['-ss', str(start_sec), '-to', str(end_sec), '-c', 'copy', file_path]
                subprocess.run(out_command)
                self.video_temp.append(file_path)

            start_sec = end_sec
        return res

    def transcript_video(self, videopath):
        # if already has subtitle file skip speech recognition
        subtitle_path = videopath.rsplit('.', 1)[0] + '.srt'
        if not os.path.isfile(subtitle_path):
            # prepare speech recognition google api
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code = 'zh-CN',
                sample_rate_hertz = 16000,
                #enable_automatic_punctuation=True,
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
            self.generate_subtitle(response.results, 4, fileName=subtitle_path)

        # embedding subtitle to video
        self.combineSubtitles(subtitle_path, videopath)
        '''
        for result in response.results:
            # First alternative is the most probable result
            alternative = result.alternatives[0]
            print(alternative.transcript)

            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time
                end_time = word_info.end_time
                print(f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}")
        '''

    def generate_subtitle(self, text_result, bin_size=3, fileName='sub.srt'):
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
                end_microsec = start_microsec + bin_size
                
                # for last word of result
                last_word_end_sec = result.alternatives[0].words[-1].end_time.seconds
                last_word_end_microsec = result.alternatives[0].words[-1].end_time.microseconds * 0.001
                
                # bin transcript
                transcript = result.alternatives[0].words[0].word
                
                index += 1 # subtitle index

                for i in range(len(result.alternatives[0].words) - 1):
                    try:
                        word = result.alternatives[0].words[i + 1].word
                        word_start_sec = result.alternatives[0].words[i + 1].start_time.seconds
                        word_start_microsec = result.alternatives[0].words[i + 1].start_time.microseconds * 0.001 # 0.001 to convert nana -> micro
                        word_end_sec = result.alternatives[0].words[i + 1].end_time.seconds
                        word_end_microsec = result.alternatives[0].words[i + 1].end_time.microseconds * 0.001

                        if word_end_sec < end_sec:
                            transcript += word
                        else:
                            previous_word_end_sec = result.alternatives[0].words[i].end_time.seconds
                            previous_word_end_microsec = result.alternatives[0].words[i].end_time.microseconds * 0.001
                            
                            # append bin transcript
                            transcriptions.append(srt.Subtitle(index, datetime.timedelta(0, start_sec, start_microsec), datetime.timedelta(0, previous_word_end_sec, previous_word_end_microsec), transcript))
                            
                            # reset bin parameters
                            start_sec = word_start_sec
                            start_microsec = word_start_microsec
                            end_sec = start_sec + bin_size
                            end_microsec = start_microsec + bin_size
                            transcript = result.alternatives[0].words[i + 1].word
                            
                            index += 1
                    except IndexError:
                        print('Error when proccessing subtitles')
                # append transcript of last transcript in bin
                transcriptions.append(srt.Subtitle(index, datetime.timedelta(0, start_sec, start_microsec), datetime.timedelta(0, last_word_end_sec, last_word_end_microsec), transcript))
                index += 1
            except IndexError:
                print('Error when transcripting video')
        
        # turn transcription list into subtitles
        subtitles = srt.compose(transcriptions)
        with open(fileName, 'w', encoding='utf8') as f:
            f.writelines(subtitles)
        print('已生成字幕文件, 文件路径: ' + fileName)
        self.srt_temp.append(fileName)

    def combineSubtitles(self, subtitle_path='', video_path=''):
        if not video_path:
            video_path = self.src
            subtitle_path = self.src.rsplit('.', 1)[0] + '.srt'

        output_path = video_path.rsplit('.', 1)[0] + '_sub.mp4'
        self.clip_temp.append(output_path)

        if not os.path.isfile(output_path):
            print('正在嵌入字幕到视频片段中...')
            subprocess.call(['ffmpeg', '-i', video_path, '-v', 'error', '-vf', 'subtitles='+subtitle_path, output_path])
        else:
            print('带字幕视频文件已存在, 跳过')

        with open(self.video_list_file, 'a', encoding='utf8') as v_list:
            v_list.write(f"file '{os.path.basename(output_path)}'\n")

    def combineVideos(self):
        if os.path.isfile(self.video_list_file):
            output_path = os.path.dirname(self.video_list_file)
            output_fp = output_path + '/' + os.path.basename(self.src)
            print('正在将视频片段合成...')
            subprocess.run(['ffmpeg','-f','concat','-i',self.video_list_file,'-v','error','-codec','copy',output_fp])

    def clear_temp_except(self, *temp_to_keep):
        print('正在清理临时文件...')

        if 'video' not in temp_to_keep:
            for v in self.video_temp:
                if os.path.isfile(v): os.remove(v)
        if 'audio' not in temp_to_keep:
            for a in self.audio_temp:
                if os.path.isfile(a): os.remove(a)
        if 'subtitle' not in temp_to_keep:
            for s in self.srt_temp:
                if os.path.isfile(s): os.remove(s)
        if 'clip' not in temp_to_keep:
            for c in self.clip_temp:
                if os.path.isfile(c): os.remove(c)
        if 'list' not in temp_to_keep:
            if os.path.isfile(self.video_list_file): os.remove(self.video_list_file)

def main():
    vc = VideoConverter('Test/22351.mp4')
    file_list = vc.split_video(sec_limit=58)
    for f in file_list:
        vc.transcript_video(f)
    vc.combineVideos()
    vc.clear_temp_except('subtitle')
    

if __name__ == '__main__':
    main()