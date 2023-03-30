import variables
import sys
sys.path.append(f"./{variables.model_name}")
from module import grpc_bench
from module import put_data_into_sheet

result = grpc_bench.run_bench(variables.num_tasks, variables.grpc_server_address, variables.use_https)

put_data_into_sheet.put_data(variables.grpc_spreadsheet_id, result, variables.num_tasks)