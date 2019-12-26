# legal_utils

utils.law_to_list(path, remain_new_line=False):把文件读成列表，每一项为一条
utils.cut_law(law_list, order=None, cut_sentence=False, cut_penalty=False, stop_words_filtered=True):输入为法条的列表。把每一法条分为序号、罪名、分词后的案情、词数
utils中其他函数自行理解

wenshu_processor.py:如__main__里的流程，威科先行爬下来的数据如下所示，
![image](https://github.com/SkyErnest/legal_utils/blob/master/image/original_file.PNG)
read_all()的输入结果如下图所示
![image](https://github.com/SkyErnest/legal_utils/blob/master/image/Capture0.PNG)
![image](https://github.com/SkyErnest/legal_utils/blob/master/image/Capture.PNG)
![image](https://github.com/SkyErnest/legal_utils/blob/master/image/Capture2.PNG)
![image](https://github.com/SkyErnest/legal_utils/blob/master/image/Capture3.PNG)
