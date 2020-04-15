import logging
import os
import shutil
import sys
import tempfile

from pyflink.table import BatchTableEnvironment, EnvironmentSettings
from pyflink.table.descriptors import FileSystem, OldCsv, Schema
from pyflink.table.types import DataTypes
from pyflink.table.udf import udf


def word_count():
    environment_settings = EnvironmentSettings.new_instance().in_batch_mode().use_blink_planner().build()
    t_env = BatchTableEnvironment.create(environment_settings=environment_settings)

    # register Results table in table environment
    tmp_dir = tempfile.gettempdir()
    result_path = tmp_dir + '/result'
    if os.path.exists(result_path):
        try:
            if os.path.isfile(result_path):
                os.remove(result_path)
            else:
                shutil.rmtree(result_path)
        except OSError as e:
            logging.error("Error removing directory: %s - %s.", e.filename, e.strerror)

    logging.info("Results directory: %s", result_path)

    # we should set the Python verison here if `Python` not point
    t_env.get_config().set_python_executable("python3")

    t_env.connect(FileSystem().path(result_path)) \
        .with_format(OldCsv()
                     .field_delimiter(',')
                     .field("city", DataTypes.STRING())
                     .field("sales_volume", DataTypes.BIGINT())
                     .field("sales", DataTypes.BIGINT())) \
        .with_schema(Schema()
                     .field("city", DataTypes.STRING())
                     .field("sales_volume", DataTypes.BIGINT())
                     .field("sales", DataTypes.BIGINT())) \
        .register_table_sink("Results")

    @udf(input_types=DataTypes.STRING(), result_type=DataTypes.ARRAY(DataTypes.STRING()))
    def split(input_str: str):
        return input_str.split(",")

    @udf(input_types=[DataTypes.ARRAY(DataTypes.STRING()), DataTypes.INT()], result_type=DataTypes.STRING())
    def get(arr, index):
        return arr[index]

    t_env.register_function("split", split)
    t_env.register_function("get", get)

    t_env.get_config().get_configuration().set_string("parallelism.default", "1")

    data = [("iPhone 11,30,5499,Beijing",),
            ("iPhone 11 Pro,20,8699,Guangzhou",),
            ("MacBook Pro,10,9999,Beijing",),
            ("AirPods Pro,50,1999,Beijing",),
            ("MacBook Pro,10,11499,Shanghai",),
            ("iPhone 11,30,5999,Shanghai",),
            ("iPhone 11 Pro,20,9999,Shenzhen",),
            ("MacBook Pro,10,13899,Hangzhou",),
            ("iPhone 11,10,6799,Beijing",),
            ("MacBook Pro,10,18999,Beijing",),
            ("iPhone 11 Pro,10,11799,Shenzhen",),
            ("MacBook Pro,10,22199,Shanghai",),
            ("AirPods Pro,40,1999,Shanghai",)]
    t_env.from_elements(data, ["line"]) \
        .select("split(line) as str_array") \
        .select("get(str_array, 3) as city, "
                "get(str_array, 1).cast(LONG) as count, "
                "get(str_array, 2).cast(LONG) as unit_price") \
        .select("city, count, count * unit_price as total_price") \
        .group_by("city") \
        .select("city, "
                "sum(count) as sales_volume, "
                "sum(total_price) as sales") \
        .insert_into("Results")

    t_env.execute("word_count")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    word_count()
