import pickle, os, json, re


def read_all(path):
    dir_list = os.listdir(path)
    facts, reasons, res = [], [], []
    for each in dir_list:
        cur_path = os.path.join(path, each)
        file_list = os.listdir(cur_path)
        for file in file_list:
            with open(os.path.join(cur_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if any(a not in data['additionalFields'] for a in ['caseDetail', 'judgmentReason', 'judgmentResult']):
                    continue
                if '二审' in data['additionalFields']['caseDetail']:
                    continue
                facts.append(data['additionalFields']['caseDetail'])
                reasons.append(data['additionalFields']['judgmentReason'])
                res.append(data['additionalFields']['judgmentResult'])

    return facts, reasons, res


def split_fact(fact):
    res = ''
    same_as_declarartion = re.compile(r'(查明|认定)(.*)(与|同)(.*)一致').search(fact) != None

    if any(a in fact for a in ['事实如下', '如下事实', '如下情况', '情况如下']):
        res += re.compile(r'((事实|情况)如下|如下(事实|情况))(.*?)(：|，)(.+?)(\r|\n|$)').search(fact).groups()[-2]
    if '经审理查明' in fact and not same_as_declarartion:
        res += re.compile(r'经审理查明(.*?)(：|，)(.+?)(\r|\n|$)').search(fact).groups()[-2]
    tmp = re.compile(r'(起诉称：|事实(与|和)理由：)(.+?)(\r|\n|$)').search(fact)
    if tmp != None and (same_as_declarartion or res == ''):
        res += tmp.groups()[-2]
    # if '经审理查明，' in fact:
    #     res+=re.compile(r'经审理查明，(.*)\r').search(fact)
    # res=re.compile(r'(认定事实如下(：|，)|经审理查明，|((事实(与|和)理由：).*查明|认定.*与|同.*一致)\r)').split(fact)[-1].strip()
    tmp = re.compile(r'(另|又|再|还|同时)查明，(.+?)(\r|\n|$)').findall(fact)
    if tmp != []:
        for each in tmp:
            res += each[1]
    return res.strip()


def split_reason(reason):
    reason = reason[-300:].replace('、《', '，《').replace('以及', '').replace('和《', '，《').replace('相关法律法规', '').replace(
        '、最高人民法院', '，最高人民法院').replace('的规定，《', '，《').replace('规定，《', '，《')
    res = []
    tmp = re.compile(r'依(照|据)(《.+?)(的|之)*(相关)*规定，').findall(reason)
    for each in tmp:
        if '《' in each[1] and '》' in each[1]:
            cites = each[1]
            # if '依照' in cites or '依据' in cites:
            #     cites=re.compile(r'(依照|依据)+(.+?)$').search(cites).group(2)
            cites = cites.split('，')
            for a in cites:
                if '《' not in a or '》' not in a or (a[-1] != '款' and a[-1] != '条' and a[-1] != '项'):
                    if a == '':
                        del a
                    else:
                        cites = []
            res += cites
    return res


def split_result(result):
    res = re.compile(r'([零一二三四五六七八九十]|[0-9])+(、|，)(.*?)(;|；|。)').findall(result)
    if res != []:
        return list(zip(*res))[2]


def split_plea(fact):
    fact = fact.replace(',', '，')
    tmp = re.compile(r'(诉讼请求|诉称)(.*?)(，|：)(.+?)(\r|事实(与|和)理由)').search(fact)
    if tmp != None:
        pleas = re.compile(r'([零一二三四五六七八九十]|[0-9])+(、|，)(.*?)(;|；|。)').findall(tmp.group(4))
        if pleas != []:
            return list(zip(*pleas))[2]


if __name__ == '__main__':
    facts, reasons, res = read_all('data/机动车事故')
    facts, reasons, res = map(lambda x: list(map(lambda a: a.replace(',', '，').replace(':', '：'), x)),
                              [facts, reasons, res])
    pleas = [split_plea(fact) for fact in facts]
    facts = [split_fact(fact) for fact in facts]
    reasons = [split_reason(reason) for reason in reasons]
    res = [split_result(result) for result in res]

    data = [each for each in zip(facts, pleas, reasons, res) if not (
            any(a == None for a in each) or any(a == [] for a in each) or any(a == '' for a in each))]
    data
