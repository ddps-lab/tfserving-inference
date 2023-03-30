import gspread
from oauth2client.service_account import ServiceAccountCredentials

def open_sheet(spreadsheet_key):
  scope = [
      'https://spreadsheets.google.com/feeds'
  ]
  cred_json_file_name = './sheet_credential.json'
  credentials = ServiceAccountCredentials.from_json_keyfile_name(cred_json_file_name,scope)
  gc = gspread.authorize(credentials)
  doc = gc.open_by_key(spreadsheet_key)
  return doc