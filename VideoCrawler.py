from VideoConverter import VideoConverter
from bs4 import BeautifulSoup as BS
from random import uniform
import webbrowser
import requests
import os, time

class Video:
    def __init__(self, name, link, parent_dir, is_mp4):
        self.name = name
        self.link = link
        self.parent_dir = parent_dir
        self.is_mp4 = is_mp4

class VideoSet:
    def __init__(self, video_set=[]):
        self.videos = video_set

    def setVideos(self, video_set):
        self.videos = video_set

    def write_to_file(self, dest='temp.txt'):
        try:
            file = open(dest, 'w')
            for v in self.videos:
                is_mp4 = '1' if v.is_mp4 else '0'
                file.write(v.name + ',' + v.link + ',' + v.parent_dir + ',' + is_mp4 + '\n')
            #file.writelines(v.name + ',' + v.link + ',' + v.parent_dir + ',' + '1' if v.is_mp4 else '0' + '\n' for v in self.videos)
            file.close()
        except IOError:
            print('写入视频资源链接缓存文件时错误: ')

    def read_from_file(self, src='temp.txt'):
        try:
            file = open(src, 'r')
            for line in file.readlines():
                v_line = line.strip().split(',')
                print(v_line)
                video = Video(v_line[0], v_line[1], v_line[2], True if v_line[3] == '1' else False)
                self.videos.append(video)
            file.close()
            return self
        except IOError:
            print('读取视频资源缓存文件失败')
            return None
        

    # Returns the number of videos that successfully downloaded
    def download(self, timer_range=(), des_dir=''):
        msg = ''
        des = ''
        downloaded = 0
        skipped = 0
        converter = VideoConverter()

        if not des_dir:
            des = os.path.join(os.getcwd(), 'Output')
            if not os.path.isdir(des): os.mkdir(des)

        # check input
        if not timer_range or not len(timer_range) == 2 or not isinstance(timer_range[0], int) or not isinstance(timer_range[1], int) or not timer_range[1] > timer_range[0]:
            msg = '等待时间区间为空或格式错误'
            print(msg)
            timer_range = (0,1)

        if len(self.videos) < 1:
            msg = '没有要下载的媒体文件'
            print(msg)
            return 0, msg

        # download video from current list one by one, with a random wait time
        for vid_idx in range(len(self.videos)):
            random_wait_time = 0 if vid_idx == 0 else uniform(timer_range[0], timer_range[1])
            video = self.videos[vid_idx]
            save_path = des + '/' + video.parent_dir
            if not os.path.isdir(save_path):
                print('Creating directories')
                os.makedirs(save_path)

            filePath = save_path + '/' + video.name
            if not os.path.isfile(filePath):
            
                print('即将在' + str(random_wait_time) + '秒后下载: ' + video.name + ' 到' + save_path)
                time.sleep(random_wait_time)

                # Finished waiting, start downloading
                with open(filePath, 'wb') as video_file:
                    video_file.write(requests.get(video.link).content)
                    downloaded += 1

                # finished downloading, if it is not in mp4 format, convert it
                if not video.is_mp4:
                    for retry in range(3):
                        if os.path.isfile(filePath):
                            success = converter.convertTo(filePath, 'mp4')
                            break
                        else:
                            print('等待文件下载中...')
                            time.sleep(10)
                    if not success:
                        print('视频转码失败')

            else:
                skipped += 1

        msg = '成功下载 ' + str(downloaded) + '/' + str(len(self.videos)) + ' 个视频文件' + ('' if skipped == 0 else (', 跳过 ' + str(skipped) + ' 个文件'))
        print(msg)
        return downloaded, msg

class VideoCrawler:
    def __init__(self, site_url, parser='html.parser'):
        self.url = site_url
        self.parser = parser
        self.catalog_name = ''

    '''
    Make soup from specified url
    '''
    def __request_page(self, url):
        result_msg = 'Getting web page data from \"' + url + '\" successful. ' 
        response = requests.get(url)
        if not response:
            result_msg = 'Fail to retrive data from \"' + url + '\". '
            return None
        return BS(response.text, self.parser), result_msg

    def __crawl(self, url='', tag='', attrs={}, retry_limit=2):
        url = self.url if not url else url

        for i in range(retry_limit):
            soup, msg = self.__request_page(url)
            #print(msg)
            if soup:
                target = soup.find_all(tag, attrs=attrs)
                if target:
                    return target
                else:
                    # open browser to manully pass recaptcha
                    res = webbrowser.open(url, new=1)
                    a = input('请手动通过验证后再按[回车]继续')  
        return None

    def __lele_getAllChapters(self):
        res = []
        # Get page main container
        chapters_container = self.__crawl(tag='div', attrs={'class':'catalog'})
        # Get Subject name
        if chapters_container:
            catalog_name = chapters_container[0].findChild('div', attrs={'class':'catalog_name'})
            if catalog_name and catalog_name.string:
                self.catalog_name = catalog_name.string.split('：', 1)[1]

            chapters_links = chapters_container[0].findChildren(
                lambda line:line.name == 'a' and line.has_attr('href') and line.string
            )

            chapter_name = ''
            for link in chapters_links:
                if '乐乐课堂版' in link.string:
                    chapter_name = link.string[5:]
                else:
                    if not link['href'] == 'javascript:;':
                        # 基于乐乐网页构造，视频链接为主页链接替换而来. 例 index20-2-170870-1.shtml -> vs20-2-170870.shtml
                        if 'index' in link['href']:
                            vs_link = 'vs' + link['href'][5:].rsplit('-', 1)[0] + '.shtml'
                            # （网页链接，用于保存本地的链接） 例: ('vs20-2-170870.shtml', '初中数学/初一上学期/概率')
                            res.append((vs_link, self.catalog_name + '/' + chapter_name + '/' + link.string))
        return res

    def lele_getAllVideoLinks(self, v_link_prefix=''):
        prefix = v_link_prefix if len(v_link_prefix) > 0 else self.url.rsplit('/', 1)[0] + '/'
        res = []
        v_set = VideoSet(res)
        cache_path = os.path.join(os.getcwd(), 'temp.txt')

        # get from local cache if has
        if os.path.isfile(cache_path):
            cache = v_set.read_from_file(cache_path)
            if cache: return cache

        # from each course, search for video container
        for (web_link, local_link) in self.__lele_getAllChapters():
            course_link = prefix + web_link

            video_container_soup = self.__crawl(course_link, 'div', {'class':'category_videos'})
            video_container_links = video_container_soup[0].findChildren(
                lambda line:line.name == 'a' and line.has_attr('href')
            )
            
            # from each video container, get each video link, and then crawl it's data to find the desired video file link
            for video_url in video_container_links:
                full_video_link = prefix + video_url['href']

                video_soup = self.__crawl(full_video_link, 'div', {'class':'play_box'})
                #print(len(video_soup))
                #print(video_soup[0])
                link = ''
                is_mp4 = False
                script_line = video_soup[0].findChild('script')
                v_links = script_line.string.split('"setMedia", {', 1)[1].split('poster:', 1)[0].strip().split(',')

                for v_l in v_links:
                    if 'm4v' in v_l:
                        link = v_l.split('\"')[1]
                        is_mp4 = True
                        break
                    else:
                        link = v_l.split('\"')[1]
                        is_mp4 = False

                print('Found video in: ' + link)

                file_name = link.rsplit('/', 1)[1]
                video = Video(file_name, link, local_link, is_mp4)
                res.append(video)

        # save cache
        v_set.setVideos(res)
        v_set.write_to_file(cache_path)

        return v_set

def test():
    vc = VideoCrawler('https://www.leleketang.com/kingkey/kb/index30-2-176357-1.shtml')
    ret, msg = vc.lele_getAllVideoLinks().download((5,20))

if __name__ == '__main__':
    test()