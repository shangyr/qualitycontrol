# -*- coding:utf-8 -*-
from .general_project import GeneralProject
from .chief_complaint import ChiefComplaint
from .present_illness import PresentIllness
from .past_illness import PastIllness
from .physical_examination import PhysicalExamination
from .auxiliary_inspection import AuxiliaryInspection
from .diagnosis import Diagnosis
from .signature import Signature
from .item_veto import ItemVeto
from .medical_record import MedicalRecord
from .disease_predict import DiseasePredict

from ..extract_entity import MatchExtract
from ..extract_entity import NERExtract
from ..extract_entity import SpacyExtract

from ..knowledge_graph import ChineseGraph

from ..typos_corrector import Corrector

class EMRQualityInspectionAlgorithmapi:
    def __init__(self, entity_dict_path, ner_model_path, kg_graph_path, diagnosis_path, corrector_path, acute_illness_path, body_path) -> None:
        match_model = MatchExtract(entity_dict_path)
        ner_model = NERExtract(ner_model_path)
        spacy_model = SpacyExtract()
        graph_model = ChineseGraph(kg_graph_path, match_model)
        corrector_model = Corrector(corrector_path)
        self._general_project_predictor = GeneralProject()
        self._chief_complaint_predictor = ChiefComplaint(match_model, ner_model,spacy_model,graph_model,corrector_model)
        self._present_illness_predictor = PresentIllness(match_model, ner_model,spacy_model,graph_model,corrector_model)
        self._past_illness_predictor = PastIllness(match_model, ner_model,spacy_model,graph_model,corrector_model)
        self._physical_examination_predictor = PhysicalExamination(match_model,ner_model,spacy_model,graph_model,corrector_model)
        self._auxiliary_inspection_predictor = AuxiliaryInspection(match_model, ner_model,spacy_model,graph_model)
        self._diagnosis_predictor = Diagnosis(match_model, ner_model, spacy_model, graph_model)
        self._signature_predictor = Signature()
        self._item_veto_predictor = ItemVeto(match_model, ner_model, spacy_model, graph_model,diagnosis_path)
        self._medical_record_predictor = MedicalRecord(match_model, ner_model, spacy_model, graph_model, acute_illness_path, body_path)
        self._disease_predict_predictor = DiseasePredict(diagnosis_path)

    def _general_project(self,emr, item, parm):
        """
        一般项目（10分）
        1.包括患者姓名、性别、年龄、工作单位或住址。
        2.每次就诊填写就诊日期及科别。
        3.代开药：姓名、身份证、电话和原因四项
        4.这里可能会增加药敏史必填。
        每缺一项扣2分。
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._general_project_predictor.check(emr, item, parm)
    
    def _chief_complaint(self, emr, item, parm):
        """
        主诉（10分）
        3.初诊：主要症状(或体征)及持续的时间。不同专科或不同病种按初诊书写。
        4.复诊：同一专科诊断明确，复诊时可写“病史同前”或“XXX疾病复诊”。
        初诊病历无描述症状（体征）、持续时间各扣5分，描述有缺陷每处扣2分。
        无主诉内容扣10分。
        描述有缺陷：主诉不超过20个字符。（还有就是没用专业医学术语）
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """

        return self._chief_complaint_predictor.check(emr, item, parm)

    def _present_illness(self, emr, item, parm):
        """
        现病史 (15分)
        5.初诊：记录本次疾病起病日期和主要症状，简要发病经过和就诊前诊治情况及有关的鉴别诊断资料。
        6.复诊：上次诊疗后的病情变化和治疗反应，未确诊病例有必须的鉴别诊断资料的补充。
        初诊病历不能反应疾病的主要症状等特征、无发病经过各扣5分。
        复诊病历无描述病情变化、治疗反应各扣5分。
        无现病史内容或直接复制主诉内容扣15分。
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._present_illness_predictor.check(emr, item, parm)
    
    def _past_illness(self, emr, item, parm):
        """
        既往史及其它病史 (10分)
        7.记录重要的或与本病相关的既往史。
        8.记录与诊治有关的药敏史、个人史（呼吸科肺癌病人如抽烟等）、婚育史、月经史、家族史等。
        缺重要既往史记录等扣5分。
        专科相关疾病，如肾病、心血管专科需要写高血压疾病。如遗传疾病肿瘤是否有家族历史。
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._past_illness_predictor.check(emr, item, parm)
    
    def _physical_examination(self, emr, item, parm):
        """
        体格检查（医生临床接触听诊器、电筒等等) (15分)

        9.初诊：一般情况，阳性体征及有助于鉴别诊断的阴性体征。
        10.复诊：重点记录原来阳性体征的变化和新发现阳性体征。

        漏一项阳性体征（异常的）扣5分，漏主要阴性体征（正常的）扣3分，体征未按要求描述扣2分。
        无体格检查内容扣15分。

        根据专科和疾病相关进行体检。
        阳性体征（根据疾病和诊断相关的体征）指导诊断，阴性体征（根据鉴别体征相关的，如肾病和肾炎鉴别）是排出诊断。
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._physical_examination_predictor.check(emr, item, parm)
    
    def _auxiliary_inspection(self, emr, item, parm):
        """
        辅助检查 (10分)
        11.与本次疾病相关的检验检查结果，应分类按检查时间顺序记录。
        12.如在其他医疗机构所作检查，应当写明该机构名称。

        未记录与本次疾病相关的重要检查检验结果扣5分。
        检验检查结果未分类整理，记录混乱扣2分。

        去其他地方做，抽血拍片等检查。主要是近期（一个月）疾病相关的检查报告出来但未写。
        （对前面近期的病历写过，之前写过这里可以不写）
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._auxiliary_inspection_predictor.check(emr, item, parm)

    def _diagnosis(self, emr, item,parm):
        """
        诊断 (10分)

        13.诊断应符合临床诊疗规范，公知公认的才可写英文缩写。
        14.诊断为多项时，应当主次分明，主要诊断在前，其他诊断在后。
        15.对不能明确诊断的待查病例应列出可能性最大的诊断。
        诊断书写不规范扣5分。
        待查病历未列出可能诊断扣2分。
        主要诊断与主诉、现病史等描述不一致扣10分。
        待查病历：如发热腹痛的原因，如写出发热是由病毒性感冒？肺炎？
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._diagnosis_predictor.check(emr,item,parm)
    
    def _signature(self, emr, item, parm):
        """
        签名（5分)

        20.经治医师签全名。
        无经治医师签名扣5分。
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._signature_predictor.check(emr, item, parm)

    def _disease_predict(self,emr):
        """
        一般项目（10分）
        1.包括患者姓名、性别、年龄、工作单位或住址。
        2.每次就诊填写就诊日期及科别。
        3.代开药：姓名、身份证、电话和原因四项
        4.这里可能会增加药敏史必填。
        每缺一项扣2分。
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._disease_predict_predictor.check(emr)
    
    def _item_veto(self, emr, item, parm):
        """
        单项否决

        21.无主诉内容。（只有·、符号也不可以）
        22.无现病史内容。
        23.无体格检查内容。
        24.主要诊断与主诉、现病史等描述不一致。
        25.无处方/处理记录内容，且无嘱托记录。
        26.用重复的内容、数字或符号填充部分病历记录。
        27.复诊病历直接复制患者既往病历记录，未体现本次诊疗的有关情况。
        符合其中一项认定为乙级病历，符合两项或以上认定为丙级病历。

        28.没有书写门诊病历记录或没有在接诊时及时完成。
        29.用重复的内容、数字或符号填充整个病历记录。
        直接认定为丙级病历。
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'vote' : str, 乙级病历 / 丙级病历 / 无否决
              'reason': str 扣分原因
            }
        """
        return self._item_veto_predictor.check(emr, item, parm)

    def _medical_record(self, emr, item, parm):
        """
        29.手术记录,手术记录创建时间超过手术结束时间24小时。
        30.手术记录,手术时间、术前诊断、术中诊断、手术名称、手术医生、麻醉方式、手术经过等缺项。
        31.手术记录，手术记录与病历记录的一致性检测。
        32.危急值记录，危急值记录创建时间超过危急值报告时间6小时。
        33.有创操作记录，有创操作记录创建时间超过操作时间24小时。
        34.有创操作记录，操作名称、操作时间、操作步骤、操作者等缺项。
        35.有创操作记录，有创操作记录与病历记录的一致性检测。

        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
        Return:
            { 'score' : int 该部分扣除的分数,
              'reason': str 扣分原因
            }
        """
        return self._medical_record_predictor.check(emr, item, parm)

    def emr_quality_inspection(self, emr, item, parm, is_veto=False):
        """
        Args:
            emr:{'姓名':'***', '性别':'*', '接诊时间':'*****',...}
            item: 需要执行的质控规则 _item_1,_item_2,....
            parm: 质控规则额外的参数
            is_veto: 是否是单项否决，可能会去掉
        Return:
            { 'level': str, 甲级病历 / 乙级病历 / 丙级病历
              'score': int, 总分
              'reason': dict {'general_projects_score':{'score': ***, 'reason': '***'},...}
            }
        """

        # 单项否决
        if is_veto:
            if item == '_item_50':
                # 疾病预测结果
                predict_ans = self._disease_predict(emr)
                parm["predict_ans"] = predict_ans
                return self._item_veto(emr, item, parm)
            return self._item_veto(emr, item, parm)

        # 预测模型
        # if item=='predict' or item=="_item_50":
        #     raw_ans = self._disease_predict(emr)
        #     # return self._disease_predict(emr)

        fields = [
            '_general_project',  # 10
            '_chief_complaint',  # 10
            '_present_illness',  # 15
            '_past_illness',  # 10
            '_physical_examination',  # 15
            '_auxiliary_inspection',  # 10
            '_diagnosis',  # 10
            '_medical_record'
        ]

        for field in fields:
            field_ans = getattr(self, field)(emr, item, parm)
            if not field_ans:
                continue
            else:
                return field_ans

