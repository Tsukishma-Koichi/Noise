import os
import json
import re
from tenacity import *
from openai import OpenAI

# ----------------------
# 智能分类模块
# ----------------------
class NoiseClassifier:
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model
        self.local_rules = [
            (self._is_steady,
             {"classification": "steady", "confidence": 0.90, "reason": "窄带且能量稳定"}),
            (self._is_non_steady,
             {"classification": "non-steady", "confidence": 0.85, "reason": "检测到时变特征"})
        ]
        self.params = {
            "bandwidth_threshold": 50,
            "energy_std_threshold": 3,
            "spike_threshold": 15
        }

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((Exception)))
    def classify(self, features):
        """分类入口（带请求预验证）"""
        try:
            self._prevalidate_request(features)
            return self._api_classify(features)
        except Exception as e:
            self._log_error(e)
            print(f"API分类失败，启用本地规则: {str(e)}")
            return self._local_classify(features)

    def _prevalidate_request(self, features):
        """请求预处理（类型检查）"""
        try:
            json.dumps(features)  # 测试序列化
        except TypeError as e:
            type_info = {k: str(type(v)) for k, v in features.items()}
            raise RuntimeError(f"特征值类型错误: {type_info}") from e

    def _api_classify(self, features):
        """API分类（增强提示工程）"""
        prompt = f"""
作为声学专家，请根据以下特征进行噪声分类（需严格遵循标准）：
{json.dumps(features, indent=2, ensure_ascii=False)}

分类标准（必须逐条检查）：
1. 稳态噪声需同时满足：
   - 主峰3dB带宽 <{self.params['bandwidth_threshold']}Hz
   - 主峰能量占比 >70%
   - 能量波动标准差 <{self.params['energy_std_threshold']}dB

2. 非稳态噪声需满足任一：
   - 存在超过{self.params['spike_threshold']}dB的瞬时能量突增
   - 主峰带宽 ≥{self.params['bandwidth_threshold'] * 2}Hz

请返回严格符合以下格式的JSON：
{{
    "classification": "steady/non-steady",
    "confidence": 置信度(0-1),
    "reason": "技术说明（需引用具体参数）"
}}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个严谨的噪声分类引擎，必须严格遵循技术标准"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )

        return self._process_response(response, features)

    def _process_response(self, response, features):
        """响应处理（正则表达式增强）"""
        content = response.choices[0].message.content

        # 使用正则表达式提取JSON内容
        if match := re.search(r'```(?:json)?\n?(.*?)```', content, re.DOTALL):
            content = match.group(1).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"无效的JSON响应: {content}") from e

        return self._validate_result(result, features)

    def _validate_result(self, result, features):
        """结果验证（增强逻辑检查）"""
        valid_classes = ["steady", "non-steady"]
        if result["classification"] not in valid_classes:
            raise ValueError(f"无效分类结果: {result['classification']}")

        # 稳态判定逻辑验证
        if result["classification"] == "steady":
            failed_conditions = []
            if features["bandwidth_3db"] >= self.params['bandwidth_threshold']:
                failed_conditions.append(f"带宽{features['bandwidth_3db']}≥阈值{self.params['bandwidth_threshold']}")
            if features["peak_energy_ratio"] < 0.7:
                failed_conditions.append(f"能量占比{features['peak_energy_ratio'] * 100:.1f}%<70%")
            if features["energy_change_std"] >= self.params['energy_std_threshold']:
                failed_conditions.append(
                    f"波动标准差{features['energy_change_std']:.1f}≥阈值{self.params['energy_std_threshold']}")

            if failed_conditions:
                raise ValueError(f"稳态判定矛盾: {', '.join(failed_conditions)}")

        return result

    def _is_steady(self, features):
        """稳态本地规则"""
        return (features["bandwidth_3db"] < self.params['bandwidth_threshold'] and
                features["peak_energy_ratio"] > 0.7 and
                features["energy_change_std"] < self.params['energy_std_threshold'])

    def _is_non_steady(self, features):
        """非稳态本地规则"""
        return (features["max_energy_change"] > self.params['spike_threshold'] or
                features["bandwidth_3db"] >= self.params['bandwidth_threshold'] * 2)

    def _local_classify(self, features):
        """本地分类（带阈值检查）"""
        for condition, result in self.local_rules:
            if condition(features):
                return result
        return {"classification": "unknown", "confidence": 0.0, "reason": "未匹配任何规则"}

    def _log_error(self, error):
        """增强错误日志"""
        log_entry = f"""
[{datetime.now().isoformat()}] 错误详情
类型: {type(error).__name__}
信息: {str(error)}
追踪:
{''.join(traceback.format_exception(type(error), error, error.__traceback__))}
        """
        with open("classification_errors.log", "a", encoding="utf-8") as f:
            f.write(log_entry)