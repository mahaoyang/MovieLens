# set env
import pandas as pd
from pyalink.alink import *
import os

resetEnv()
useLocalEnv(2)


def get_train_data(use_cache=False, all_columns=False, export_feature=False, user_action_day=7, debug_length=None, skip_label=False):
    cache_file_name = 'live_data.csv'
    interval_offset = 1
    if use_cache:
        if os.path.exists(cache_file_name):
            return pd.read_csv(cache_file_name)


def df2op(df, schema, op_type='batch'):
    # schema = "f_string string,f_long long,f_int int,f_double double,f_boolean boolean"
    if op_type == 'batch':
        op = BatchOperator.fromDataframe(df.reset_index(drop=True), schema)
        op.print()
    else:
        op = StreamOperator.fromDataframe(df.reset_index(drop=True), schema)
    return op


column = ['source', 'shop_authentication_type', 'shop_fans_count', 'label']
df_data = get_train_data(use_cache=True)[-20000:][column].fillna(0).astype('int32')

# schema of train data
schemaStr = "source bigint, shop_authentication_type bigint, shop_fans_count bigint, label string"

# prepare batch train data
trainBatchData = df2op(df_data, schemaStr)
# feature fit
labelColName = "label"
vecColName = "vec"
numHashFeatures = 30000
selectedColNames = column

categoryColNames = ['source', 'shop_authentication_type']

numericalColNames = ['shop_fans_count']

# prepare stream train data
# data = df2op(df_data[-10:], schemaStr, op_type='stream')
bootstrap_servers = '172.16.100.31:9092,172.16.100.29:9092,172.16.100.30:9092'
topic_name = 'Topic_Live_Heartbeat_Msg'
consumer_id = 'Gid_Real_Time_Live_Heartbeat_Msg'
data = KafkaSourceStreamOp() \
    .setBootstrapServers(bootstrap_servers) \
    .setTopic(topic_name) \
    .setStartupMode("LATEST") \
    .setGroupId(consumer_id)
col = ['source', 'shop_authentication_type',
       'shop_fans_count', 'label']
# data.print()
data = JsonValueStreamOp().setJsonPath(["$." + i for i in col]).setSelectedCol("message").setOutputCols(
    col).linkFrom(data).select(col)
# data.print()
# data = data.link(FilterStreamOp().setClause("label='1'"))
# data.print()
col_type = ["LONG" for i in range(len(col) - 1)]
col_type.append('STRING')
col_sql = list(zip(col, col_type))
base_cast_sql = '%s.cast(%s) as %s'
sql = ', '.join([base_cast_sql % (i[0], i[1], i[0]) for i in col_sql])
sql = 'source.cast(LONG) as source, shop_authentication_type.cast(LONG) as shop_authentication_type, shop_fans_count.cast(LONG) as shop_fans_count, label.cast(STRING) as label'
print(sql)
fTable = data.getOutputTable()
data = fTable.select(sql)
data = TableSourceStreamOp(data)
# data.print()
# StreamOperator.execute()
print()
# data = data.getOutputTable()

# it = [DataTypes.STRING() for i in range(len(col))]
# ot = [DataTypes.BIGINT() for ii in range(len(col) - 1)]
# ot.append(DataTypes.STRING())
#
#
# @udtf(input_types=it, result_types=ot)
# def f_udtf2(*args):
#     for arg in args:
#         ag = [int(iii) for iii in arg[:-1]]
#         ag.append(arg[-1])
#         yield ag
#
#
# result_types = ['BIGINT' for iii in range(len(col) - 1)]
# result_types.append('STRING')
# udfOp = UDTFStreamOp() \
#     .setFunc(val=f_udtf2) \
#     .setResultTypes(result_types) \
#     .setSelectedCols([i + '_1' for i in col]) \
#     .setOutputCols(col)
# data = udfOp.linkFrom(data)
# data.print()
# StreamOperator.execute()
col = ['source', 'shop_authentication_type', 'shop_fans_count', 'label']
# data = JsonValueStreamOp().setJsonPath(["$." + i for i in col]).setSelectedCol("message").setOutputCols(col).linkFrom(data).select(col)
col_type = ["LONG" for i in range(len(col) - 1)]
col_type.append('STRING')
col_sql = list(zip(col, col_type))
base_cast_sql = '%s.cast(%s) as %s'
sql = ', '.join([base_cast_sql % (i[0], i[1], i[0]) for i in col_sql])
print(sql)
fTable = data.getOutputTable()
stream_source = fTable.select(sql)
data = TableSourceStreamOp(stream_source)
# data.print()
# StreamOperator.execute()
# split stream to train and eval data
# spliter = SplitStreamOp().setFraction(0.5).linkFrom(data)
# train_stream_data = spliter
# test_stream_data = spliter.getSideOutput(0)
train_stream_data = test_stream_data = data
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
    .setReservedCols([labelColName, 'room_id']) \
    .setPredictionDetailCol("details") \
    .linkFrom(model, feature_pipelineModel.transform(test_stream_data))

predResult.print(refreshInterval=30, maxLimit=20)
# csvSink = CsvSinkStreamOp() \
#     .setFilePath('csv_test_s.txt')
# predResult.link(csvSink)
# if __name__ == '__main__':
#     from redis import Redis
#     @udtf(input_types=[DataTypes.STRING(), DataTypes.STRING()], result_types=[DataTypes.INT(), DataTypes.DOUBLE()])
#     def f_udtf2(*args):
#         for index, arg in enumerate(args):
#             rc = Redis()
#             rc.set(index, arg)
#     udtfStreamOp = UDTFStreamOp() \
#         .setFunc(f_udtf2) \
#         .setSelectedCols(["label", "pred"]) \
#         .setOutputCols(["index", "sepal_length"]) \
#         .linkFrom(predResult)
#     udtfStreamOp.print(refreshInterval=30, maxLimit=20)


# # ftrl eval
# EvalBinaryClassStreamOp() \
#     .setLabelCol(labelColName) \
#     .setPredictionCol("pred") \
#     .setPredictionDetailCol("details") \
#     .setTimeInterval(10) \
#     .linkFrom(predResult) \
#     .link(JsonValueStreamOp() \
#           .setSelectedCol("Data") \
#           .setReservedCols(["Statistics"]) \
#           .setOutputCols(["Accuracy", "AUC", "ConfusionMatrix"]) \
#           .setJsonPath(["$.Accuracy", "$.AUC", "$.ConfusionMatrix"])) \
#     .print(key="evaluation", refreshInterval=30, maxLimit=20)
StreamOperator.execute()
resetEnv()
