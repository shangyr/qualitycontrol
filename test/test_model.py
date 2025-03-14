# coding:utf-8 -*-
import sys
import os

PROJECT_DIR = r'C:\Users\Administrator\Desktop\qualitycontrol_service'
DIAGNOSIS_DIR = r'C:\Users\Administrator\Desktop\qualitycontrol_service\src\diagnosis'
sys.path.append(PROJECT_DIR)
sys.path.append(DIAGNOSIS_DIR)
from src.diagnosis import Predictor,ChineseALLConfig

if __name__ == '__main__':
    diagnosis_path = os.path.join(PROJECT_DIR, 'data_and_models\\diagnosis')
    cfg = ChineseALLConfig(data_dir=diagnosis_path)
    p = Predictor(cfg)
    data = [
        {
            "emr_id":'123456',
            "doc":"患者10余年前无明显诱因下出现口干多饮，每日饮开水2瓶，同时体重逐渐下降，三月来体重下降约5kg，伴乏力，无视物模糊，无手足麻木，无泡沫尿，无血尿，无尿急尿痛，无心慌手抖，无怕热出汗，遂去当地医院就诊，空腹血糖偏高（具体不详），诊断为\"2型糖尿病\"，予“二甲双胍”、“格列齐特缓释片”口服降糖治疗后好转。期间未正规检测血糖，具体治疗不详，自觉口干多饮多尿多食逐渐加重。1周前患者自觉上述症状加剧，伴乏力，泡沫尿明显，双足麻木，无视物模糊，无血尿，无尿急尿痛，无心慌手抖，无怕热出汗等不适。今至我院门诊就诊，查尿微量总蛋白：238.0mg/L，尿微量白蛋白：104.1mg/L。尿酮体：1+mmol/L，葡萄糖：4+mmol/L，糖化血红蛋白：13.9%。现为求进一步诊治，门诊拟\"消渴类病\"\"2型糖尿病\"收住入院治疗。近来精神可，纳可，寐安，小便如上诉，大便尚调，近来体重无明显下降。",
            "label":[]
        }
    ]
    raw_ans = p.predict(data) # 调用一次执行一次
    ans = []
    for key in raw_ans[0]['predict_score']:
        ans.append((key,raw_ans[0]['predict_score'][key]))
    ans.sort(key=lambda x:x[1],reverse=True)
    ans = ans[:10]

    print(ans)
