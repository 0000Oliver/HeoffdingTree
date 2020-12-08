from ht.splitmetric import SplitMetric
from core import utils
import math

class InfoGainSplitMetric(SplitMetric):
    """The Info Gain split metric."""
    def __init__(self, min_frac_weight_for_two_branches):
        self._min_frac_weight_for_two_branches = min_frac_weight_for_two_branches

    # dist[(class_value,mass)]
    def evaluate_split(self, pre_dist, post_dist):
        #pre_dist {'<=50K': <ht.weightmass.WeightMass object at 0x114b5ab00>, '>50K': <ht.weightmass.WeightMass object at 0x114b5ad30>}
        #post_dist [{'<=50K': <ht.weightmass.WeightMass object at 0x114b4a978>, '>50K': <ht.weightmass.WeightMass object at 0x114b4ad30>},
        # {'<=50K': <ht.weightmass.WeightMass object at 0x114b4ac18>}, {'<=50K': <ht.weightmass.WeightMass object at 0x114b4a438>},
        # {'<=50K': <ht.weightmass.WeightMass object at 0x114b4ad68>}, {'<=50K': <ht.weightmass.WeightMass object at 0x114b4a6d8>}]
        '''
        给出分裂前的权重和分裂后的权重，计算信息增益，评估该分裂值。
        :param pre_dist:
        :param post_dist:
        :return:
        '''
        pre = []
        for class_value, mass in pre_dist.items():
            pre.append(pre_dist[class_value].weight)
        pre_entropy = utils.entropy(pre)

        dist_weights = []
        total_weight = 0.0
        for i in range(len(post_dist)):
            dist_weights.append(self.sum(post_dist[i]))
            total_weight += dist_weights[i]

        frac_count = 0
        for d in dist_weights:
            if d / total_weight > self._min_frac_weight_for_two_branches:#检查每个分支分支的占比是否大于给定的最低比例
                frac_count += 1

        if frac_count < 2:
            return -math.inf #如果小于二说明 其中一个分支样本数过小，直接返回最小值淘汰这个分裂值

        post_entropy = 0
        for i in range(len(post_dist)):
            d = post_dist[i]
            post = []
            for class_value, mass in d.items():
                post.append(mass.weight)
            post_entropy += dist_weights[i] * utils.entropy(post)

        if total_weight > 0:
            post_entropy /= total_weight

        return pre_entropy - post_entropy

    def get_metric_range(self, pre_dist):
        num_classes = len(pre_dist)
        if num_classes < 2:
            num_classes = 2

        return math.log2(num_classes)