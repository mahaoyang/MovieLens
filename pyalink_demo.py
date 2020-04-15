# from pyflink.dataset import ExecutionEnvironment
# exec_env = ExecutionEnvironment.get_execution_environment()
# exec_env.set_python_requirements("req.txt", "/mnt/c/Users/99263/PycharmProjects/MovieLens/")
def d2op(df, schema):
    # schema = "f_string string,f_long long,f_int int,f_double double,f_boolean boolean"
    # op = BatchOperator.fromDataframe(df, schema)
    # op.print()
    # return op
    for i in range(len(df)):
        df_row = df.loc[i:i]
    op = StreamOperator.fromDataframe(df, schema)
    op.print()
    StreamOperator.execute()
    return op


def op2d(source):
    df = source.collectToDataframe()
    return df


from pyalink.alink import *

resetEnv()
useLocalEnv(12, config=None)
schema = "age bigint, workclass string, fnlwgt bigint, education string, education_num bigint, marital_status string, occupation string,  relationship string, race string, sex string, capital_gain bigint,  capital_loss bigint, hours_per_week bigint, native_country string, label string"
# adult_batch = CsvSourceBatchOp().setFilePath("https://alink-release.oss-cn-beijing.aliyuncs.com/data-files/adult_train.csv").setSchemaStr(schema)
adult_batch = CsvSourceBatchOp().setFilePath("https://alink-release.oss-cn-beijing.aliyuncs.com/data-files/adult_test.csv").setSchemaStr(schema)
adult_stream = op2d(adult_batch)
adult_stream = d2op(adult_stream, schema)


categoricalColNames = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
numerialColNames = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
onehot = OneHotEncoder().setSelectedCols(categoricalColNames).setOutputCols(["output"]).setReservedCols(numerialColNames + ["label"])
assembler = VectorAssembler().setSelectedCols(["output"] + numerialColNames).setOutputCol("vec").setReservedCols(["label"])
pipeline = Pipeline().add(onehot).add(assembler)
logistic = LogisticRegression().setVectorCol("vec").setLabelCol("label").setPredictionCol("pred").setPredictionDetailCol("detail")
model = pipeline.add(logistic).fit(adult_batch)
predictBatch = model.transform(adult_batch)
predictBatch.print()
predictStream = model.transform(adult_stream)
predictStream.print()
StreamOperator.execute()
metrics = EvalBinaryClassBatchOp().setLabelCol("label").setPredictionDetailCol("detail").linkFrom(predictBatch).collectMetrics()
print("AUC:", metrics.getAuc())
print("KS:", metrics.getKs())
print("PRC:", metrics.getPrc())
print("Precision:", metrics.getPrecision())
print("Recall:", metrics.getRecall())
print("F1:", metrics.getF1())
print("ConfusionMatrix:", metrics.getConfusionMatrix())
print("LabelArray:", metrics.getLabelArray())
print("LogLoss:", metrics.getLogLoss())
print("TotalSamples:", metrics.getTotalSamples())
print("ActualLabelProportion:", metrics.getActualLabelProportion())
print("ActualLabelFrequency:", metrics.getActualLabelFrequency())
print("Accuracy:", metrics.getAccuracy())
print("Kappa:", metrics.getKappa())
