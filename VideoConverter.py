import cv2, os, datetime
import ffmpeg, srt, subprocess
from moviepy import editor
from google.cloud import speech

class VideoConverter:
    def __init__(self, src='', des='Result.mp4'):
        self.src = src
        self.des = des
        self.supported_video_output_format = dict([('mp4', 'h264')])

    def setSrc(self, src):
        self.src = src

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

    """
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

    # TODO: Abandon Moviepy, use FFmpeg directly instead
    # return a list of splitted video_paths
    def split_video(self, output_folder='TEMP', sec_limit=0):
        if sec_limit == 0: return [self.src]

        res = []
        full_clip = editor.VideoFileClip(self.src)

        # make sure output folder is ready
        if not os.path.isdir(output_folder): os.makedirs(os.getcwd() + '/' + output_folder)
        
        split_count = full_clip.duration / sec_limit
        if split_count <= 1:
            res.append(self.src)
            return res
        else:
            start_time = 0
            for i in range(int(split_count) + 1):
                if start_time + sec_limit <= full_clip.duration:
                    end_time = start_time + sec_limit
                    finished = False
                else:
                    end_time = full_clip.duration
                    finished = True

                src_path_arr = self.src.rsplit('.', 1)
                file_path = output_folder + src_path_arr[0] + '_' + str(i) + '.' + src_path_arr[1]
                if not os.path.isfile(file_path):
                    sub_clip = full_clip.subclip(start_time, end_time)
                    sub_clip.write_videofile(file_path)
                res.append(file_path)

                if finished:
                    break
                else:
                    start_time = end_time
        return res

    def transcript_video(self, videopath):
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
        audio_file_path = videopath.rsplit('.')[0] + '.wav'
        if not os.path.isfile:
            ffmpeg.input(videopath).output(audio_file_path, ac=1, ar='16000').run()

        with open(audio_file_path, "rb") as f:
            content = f.read()
        audio = speech.RecognitionAudio(content=content)
        response = client.recognize(request={"config": config, "audio": audio})
        sub_path = videopath.rsplit('.', 1)[0] + '.srt'
        self.generate_subtitle(response.results, 4, fileName=sub_path)
        self.combineSubtitles(sub_path, videopath)
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

    def combineSubtitles(self, subtitle_path='', video_path=''):
        if not video_path:
            video_path = self.src
            subtitle_path = self.src.rsplit('.', 1)[0] + '.srt'
        output_path = video_path.rsplit('.', 1)[0] + '_sub.mp4'
        subprocess.call(['ffmpeg', '-i', video_path, '-vf', 'subtitles='+subtitle_path, output_path])
        #ffmpeg.input(video_path).output(video_path, subtitles=subtitle_path).run()


def main():
    vc = VideoConverter('22028.mp4')
    vc
    #file_list = vc.prepareAudio()
    #vc.transcript_video('TEMP/22028_0.mp4')
    #sub = vc.generate_subtitle(resp)
    #vc.combineSubtitles('TEMP/22028_0.srt', 'TEMP/22028_0.mp4')
    #print(transcript)
    

if __name__ == '__main__':
    main()