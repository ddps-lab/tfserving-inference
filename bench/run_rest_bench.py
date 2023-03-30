import variables
import sys
sys.path.append(f"/{variables.architecture}/{variables.model_name}/bench")
import rest_bench
import put_data_into_sheet

result = rest_bench.run_bench(variables.num_tasks, variables.rest_server_address)

put_data_into_sheet.put_data(variables.rest_spreadsheet_id, result, variables.num_tasks)