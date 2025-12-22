# Source: https://modelscope.cn/datasets/aemilia/AIGCDetectionBenchmark/tree/master/AIGCDetectionBenchMark
# Reference: https://github.com/Ekko-zn/AIGCDetectBenchmark
wget --no-check-certificate https://modelscope.cn/datasets/aemilia/AIGCDetectionBenchmark/resolve/master/AIGCDetectionBenchMark/test_set.zip
unzip test_set.zip
rm test_set.zip
# Should contain an additional directory "test" after unzipping
