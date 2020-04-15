# set env
from pyalink.alink import *
import sys, os

resetEnv()
useLocalEnv(2)

# schema of train data
schemaStr = "id string, click string, dt string, C1 string, banner_pos int, site_id string, \
            site_domain string, site_category string, app_id string, app_domain string, \
            app_category string, device_id string, device_ip string, device_model string, \
            device_type string, device_conn_type string, C14 int, C15 int, C16 int, C17 int, \
            C18 int, C19 int, C20 int, C21 int"

# prepare batch train data
batchTrainDataFn = "http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-small.csv"
trainBatchData = CsvSourceBatchOp().setFilePath(batchTrainDataFn) \
    .setSchemaStr(schemaStr) \
    .setIgnoreFirstLine(True)
# feature fit
labelColName = "click"
vecColName = "vec"
numHashFeatures = 30000
selectedColNames = ["C1", "banner_pos", "site_category", "app_domain",
                    "app_category", "device_type", "device_conn_type",
                    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
                    "site_id", "site_domain", "device_id", "device_model"]

categoryColNames = ["C1", "banner_pos", "site_category", "app_domain",
                    "app_category", "device_type", "device_conn_type",
                    "site_id", "site_domain", "device_id", "device_model"]

numericalColNames = ["C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]

# prepare stream train data
wholeDataFile = "http://alink-release.oss-cn-beijing.aliyuncs.com/data-files/avazu-small.csv"
data = CsvSourceStreamOp() \
    .setFilePath(wholeDataFile) \
    .setSchemaStr(schemaStr) \
    .setIgnoreFirstLine(True)

# split stream to train and eval data
spliter = SplitStreamOp().setFraction(0.5).linkFrom(data)
train_stream_data = spliter
test_stream_data = spliter.getSideOutput(0)

# setup feature enginerring pipeline
feature_pipeline = Pipeline() \
    .add(StandardScaler() \
         .setSelectedCols(numericalColNames)) \
    .add(FeatureHasher() \
         .setSelectedCols(selectedColNames) \
         .setCategoricalCols(categoryColNames) \
         .setOutputCol(vecColName) \
         .setNumFeatures(numHashFeatures))

# fit and save feature pipeline model
FEATURE_PIPELINE_MODEL_FILE = os.path.join(os.getcwd(), "feature_pipe_model.csv")
feature_pipeline.fit(trainBatchData).save(FEATURE_PIPELINE_MODEL_FILE)

BatchOperator.execute()

# load pipeline model
feature_pipelineModel = PipelineModel.load(FEATURE_PIPELINE_MODEL_FILE)

# train initial batch model
lr = LogisticRegressionTrainBatchOp()
initModel = lr.setVectorCol(vecColName) \
    .setLabelCol(labelColName) \
    .setWithIntercept(True) \
    .setMaxIter(10) \
    .linkFrom(feature_pipelineModel.transform(trainBatchData))

# ftrl train
model = FtrlTrainStreamOp(initModel) \
    .setVectorCol(vecColName) \
    .setLabelCol(labelColName) \
    .setWithIntercept(True) \
    .setAlpha(0.1) \
    .setBeta(0.1) \
    .setL1(0.01) \
    .setL2(0.01) \
    .setTimeInterval(10) \
    .setVectorSize(numHashFeatures) \
    .linkFrom(feature_pipelineModel.transform(train_stream_data))

# ftrl predict
predResult = FtrlPredictStreamOp(initModel) \
    .setVectorCol(vecColName) \
    .setPredictionCol("pred") \
    .setReservedCols([labelColName]) \
    .setPredictionDetailCol("details") \
    .linkFrom(model, feature_pipelineModel.transform(test_stream_data))

predResult.print(key="predResult", refreshInterval=30, maxLimit=20)

# ftrl eval
EvalBinaryClassStreamOp() \
    .setLabelCol(labelColName) \
    .setPredictionCol("pred") \
    .setPredictionDetailCol("details") \
    .setTimeInterval(10) \
    .linkFrom(predResult) \
    .link(JsonValueStreamOp() \
          .setSelectedCol("Data") \
          .setReservedCols(["Statistics"]) \
          .setOutputCols(["Accuracy", "AUC", "ConfusionMatrix"]) \
          .setJsonPath(["$.Accuracy", "$.AUC", "$.ConfusionMatrix"])) \
    .print(key="evaluation", refreshInterval=30, maxLimit=20)
StreamOperator.execute()
