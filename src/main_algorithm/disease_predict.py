from ..diagnosis import Predictor, ChineseALLConfig
class DiseasePredict:
    def __init__(self, diagnosis_path:str) -> None:
        self.diagnosis_path = diagnosis_path

    def check(self, emr):
        """
        疾病预测模块
        """
        cfg = ChineseALLConfig(data_dir=self.diagnosis_path)
        p = Predictor(cfg)
        data = [
            {
                "emr_id": '123456',
                "doc": emr.get("主诉", "") + " " + \
                                emr.get("现病史", "") + " " + \
                                emr.get("既往史及其他病史", "") + " " + \
                                emr.get("体格检查", "") + " " + \
                                emr.get("辅助检查", ""),
                "label": []
            }
        ]
        raw_ans = p.predict(data)  # 调用一次执行一次
        ans = []
        for key in raw_ans[0]['predict_score']:
            ans.append((key, raw_ans[0]['predict_score'][key]))
        ans.sort(key=lambda x: x[1], reverse=True)
        ans = ans[:5]

        return ans