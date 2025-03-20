import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import jdatetime
import math
import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
import streamlit.components.v1 as components
import zoneinfo

tehran_tz = zoneinfo.ZoneInfo("Asia/Tehran")

##############################################################################
#  GOOGLE SHEETS & DATE PARSING HELPERS
##############################################################################
@st.cache_data
def create_gsheets_connection():
    service_account_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build('sheets', 'v4', credentials=creds)
    return service

def read_sheet_values(service, spreadsheet_id, sheet_name, cell_range):
    rng = f"{sheet_name}!{cell_range}"
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=rng
    ).execute()
    values = result.get('values', [])
    if not values:
        return pd.DataFrame()
    header = values[0]
    data = values[1:]
    return pd.DataFrame(data, columns=header)

def parse_input_date_str(date_str):
    s = str(date_str).strip()
    try:
        dt = datetime.datetime.strptime(s, "%Y/%m/%d")
        return dt.date()
    except:
        return None

def parse_output_date_str(date_str):
    s = str(date_str).strip()
    try:
        dt = datetime.datetime.strptime(s, "%A, %B %d, %Y at %I:%M:%S %p")
        return dt.date()
    except:
        return None

def safe_int(val):
    if val is None:
        return 0
    return 1 if str(val).strip() == "1" else 0

def compute_avg_for_weekday(input_df, target_weekday, days_interval):
    system_today = datetime.datetime.now(tehran_tz).date()
    start_date = system_today - datetime.timedelta(days=days_interval)
    mask = ((input_df["parsed_input_date"] >= start_date) &
            (input_df["parsed_input_date"] <= system_today) &
            (input_df["parsed_input_date"].apply(lambda d: d.weekday() if pd.notnull(d) else -1) == target_weekday))
    filtered = input_df[mask]
    if not filtered.empty:
        try:
            return filtered["Blank"].astype(float).mean()
        except:
            return 0
    return 0

##############################################################################
#  MODEL FORECAST HELPERS
##############################################################################
def forecast_univariate_statsmodels(model_fit, shift):
    steps_ahead = shift + 1
    fc = model_fit.forecast(steps=steps_ahead)
    return float(fc.iloc[-1])

def forecast_moving_avg(ma_dict):
    if not isinstance(ma_dict, dict):
        return np.nan
    last_vals = ma_dict.get("last_window", [])
    if len(last_vals) == 0:
        return np.nan
    return float(np.mean(last_vals))

def forecast_ts_decomp_reg(ts_tuple, X_today, shift):
    decomposition, lr = ts_tuple
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    last_trend = float(trend.dropna().iloc[-1]) if (trend is not None and not trend.dropna().empty) else 0.0
    seas_vals = seasonal.dropna().values if (seasonal is not None and not seasonal.dropna().empty) else [0.0]
    pos = shift % len(seas_vals)
    try:
        resid_pred = float(lr.predict(X_today)[0])
    except:
        resid_pred = 0.0
    return last_trend + seas_vals[pos] + resid_pred

##############################################################################
#  CUSTOM CSS & FONT
##############################################################################
def load_css():
    st.markdown(
        """
        <style>
        * { font-family: "Tahoma", sans-serif !important; }
        body { background-color: #eef2f7; direction: rtl; text-align: center; }
        .header { background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; font-size: 28px; text-align: center; }
        .scoreboard { background-color: #ecf0f1; border: 2px solid #34495e; border-radius: 5px; padding: 8px; margin-bottom: 8px; text-align: center; font-size: 16px; font-weight: bold; color: #34495e; }
        .login-box { max-width: 300px; margin: auto; padding: 40px; border: 1px solid #ddd; border-radius: 5px; background-color: #fff; color: #333; }
        </style>
        """,
        unsafe_allow_html=True
    )

##############################################################################
#  GLOBAL CONFIGURATIONS
##############################################################################
hotel_name_map = {
    "Ashrafi": "اشرفی",
    "Evin": "اوین",
    "Gandhi": "گاندی",
    "Jordan": "جردن",
    "Keshavarz": "کشاورز",
    "Koroush": "کوروش",
    "Mirdamad": "میرداماد",
    "Niloofar": "نیلوفر",
    "Nofel": "نوفل",
    "Parkway": "پارک وی",
    "Pasdaran": "پاسداران",
    "Toranj": "ترنج",
    "Valiasr": "ولیعصر",
    "Vila": "ویلا"
}

# Model configuration exactly as trained.
best_model_map = {
  "Ashrafi": ["linear_reg","random_forest","random_forest","random_forest","random_forest","random_forest","lasso_reg"],
  "Evin":    ["linear_reg","linear_reg","linear_reg","random_forest","random_forest","random_forest","random_forest"],
  "Gandhi":  ["lasso_reg","lasso_reg","holt_winters","holt_winters","holt_winters","holt_winters","holt_winters"],
  "Jordan":  ["ridge_reg","ridge_reg","lasso_reg","linear_reg","lasso_reg","linear_reg","lasso_reg"],
  "Keshavarz": ["lasso_reg","random_forest","random_forest","ridge_reg","ridge_reg","ridge_reg","ridge_reg"],
  "Koroush": ["ridge_reg","lasso_reg","ridge_reg","ridge_reg","random_forest","ridge_reg","ridge_reg"],
  "Mirdamad": ["poisson_reg","linear_reg","lasso_reg","lasso_reg","lasso_reg","lasso_reg","poisson_reg"],
  "Niloofar": ["random_forest","ridge_reg","ridge_reg","ridge_reg","ridge_reg","lasso_reg","ridge_reg"],
  "Nofel":   ["lasso_reg","random_forest","poisson_reg","lasso_reg","poisson_reg","poisson_reg","poisson_reg"],
  "Parkway": ["ridge_reg","random_forest","lasso_reg","lasso_reg","lasso_reg","lasso_reg","lasso_reg"],
  "Pasdaran": ["linear_reg","linear_reg","linear_reg","random_forest","lasso_reg","poisson_reg","poisson_reg"],
  "Toranj":  ["lasso_reg","poisson_reg","poisson_reg","poisson_reg","moving_avg","moving_avg","moving_avg"],
  "Valiasr": ["linear_reg","linear_reg","linear_reg","linear_reg","linear_reg","linear_reg","random_forest"],
  "Vila":    ["poisson_reg","lasso_reg","lasso_reg","ridge_reg","ridge_reg","lasso_reg","ridge_reg"]
}
chain_shift_models = ["linear_reg","xgboost","xgboost","xgboost"]

HOTEL_CONFIG = {
   "Ashrafi": {
     "model_prefix": "Ashrafi",
     "lag_cols": ["AshrafiN", "AshrafiS"],
     "column_order": [
        "Ramadan_dummy","Moharram_dummy","Eid_Fetr_dummy","Norooz_dummy","Sizdah-be-Dar_dummy",
        "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
        "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms","Lag10_EmptyRooms",
        "Lag11_EmptyRooms","Lag12_EmptyRooms","WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Evin": {
     "model_prefix": "Evin",
     "lag_cols": ["Evin"],
     "column_order": [
       "Ramadan_dummy","Shabe_Ghadr_dummy","Norooz_dummy","Sizdah-be-Dar_dummy",
       "Esfand_dummy","Last 5 Days of Esfand_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Gandhi": {
     "model_prefix": "Gandhi",
     "lag_cols": ["Ghandi1", "Ghandi2"],
     "column_order": [
       "Ramadan_dummy","Moharram_dummy","Shabe_Ghadr_dummy","Eid_Fetr_dummy","Norooz_dummy",
       "Sizdah-be-Dar_dummy","Yalda_dummy","Last 5 Days of Esfand_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Jordan": {
     "model_prefix": "Jordan",
     "lag_cols": ["JordanN", "JordanS"],
     "column_order": [
       "Ramadan_dummy","Moharram_dummy","Eid_Fetr_dummy","Norooz_dummy","Sizdah-be-Dar_dummy",
       "Esfand_dummy","Last 5 Days of Esfand_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Keshavarz": {
     "model_prefix": "Keshavarz",
     "lag_cols": ["Keshavarz"],
     "column_order": [
       "Ramadan_dummy","Eid_Fetr_dummy","Norooz_dummy","Sizdah-be-Dar_dummy","Last 5 Days of Esfand_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Koroush": {
     "model_prefix": "Kourosh",
     "lag_cols": ["Kourosh"],
     "column_order": [
       "Eid_Fetr_dummy","Sizdah-be-Dar_dummy","Yalda_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6",
       "Hol_holiday","Hol_none","Hol_religious_holiday"
     ]
   },
   "Mirdamad": {
     "model_prefix": "Mirdamad",
     "lag_cols": ["Mirdamad1", "Mirdamad2"],
     "column_order": [
       "Moharram_dummy","Arbain_dummy","Shabe_Ghadr_dummy","Sizdah-be-Dar_dummy","Esfand_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms","Lag10_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Niloofar": {
     "model_prefix": "Niloofar",
     "lag_cols": ["NiloofarJacuzi", "Niloofar2R", "Niloofar104"],
     "column_order": [
       "Eid_Fetr_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Nofel": {
     "model_prefix": "Nofel",
     "lag_cols": ["Nofel1", "Nofel2"],
     "column_order": [
       "Ramadan_dummy","Shabe_Ghadr_dummy","Norooz_dummy","Sizdah-be-Dar_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms","Lag10_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Parkway": {
     "model_prefix": "Parkway",
     "lag_cols": ["Parkway70", "Parkway80", "Parkway105", "Parkway6"],
     "column_order": [
       "Ramadan_dummy","Moharram_dummy","Eid_Fetr_dummy","Norooz_dummy","Sizdah-be-Dar_dummy","Yalda_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms","Lag6_EmptyRooms",
       "Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms","Lag10_EmptyRooms","Lag11_EmptyRooms","Lag12_EmptyRooms","Lag13_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Pasdaran": {
     "model_prefix": "Pasdaran",
     "lag_cols": ["Pasdaran1", "Pasdaran2"],
     "column_order": [
       "Ashoora_dummy","Norooz_dummy","Sizdah-be-Dar_dummy","Last 5 Days of Esfand_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Toranj": {
     "model_prefix": "Toranj",
     "lag_cols": ["Toranj"],
     "column_order": [
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms","Lag10_EmptyRooms","Lag11_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6","Hol_holiday","Hol_none"
     ]
   },
   "Valiasr": {
     "model_prefix": "Valiasr",
     "lag_cols": ["ValiasrN", "ValiasrS"],
     "column_order": [
       "Ramadan_dummy","Shabe_Ghadr_dummy","Norooz_dummy","Sizdah-be-Dar_dummy","Last 5 Days of Esfand_dummy",
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
     ]
   },
   "Vila": {
     "model_prefix": "Vila",
     "lag_cols": ["VilaA", "VilaB"],
     "column_order": [
       "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
       "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms",
       "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6","Hol_holiday","Hol_none"
     ]
   }
}

##############################################################################
#  READ MAIN DATA FROM GOOGLE SHEETS (CACHED)
##############################################################################
@st.cache_data
def read_main_dfs():
    service = create_gsheets_connection()
    SPREADSHEET_ID = "1LI0orqvqci1d75imMfHKxZ512rUUlpA7P1ZYjV-uVO0"
    input_df = read_sheet_values(service, SPREADSHEET_ID, "Input", "A1:ZZ10000")
    input_df["Date"] = input_df.iloc[:, 3]
    input_df["Blank"] = input_df.iloc[:, 2]
    input_df["Hold"] = input_df.iloc[:, 1]
    input_df["Week Day"] = input_df.iloc[:, 4]
    input_df["parsed_input_date"] = input_df["Date"].apply(parse_input_date_str)
    output_df = read_sheet_values(service, SPREADSHEET_ID, "Output", "A1:ZZ10000")
    output_df["parsed_output_date"] = output_df["Date"].apply(parse_output_date_str)
    return input_df, output_df

##############################################################################
#  PREDICTION FUNCTIONS FOR HOTEL & CHAIN MODELS
##############################################################################
def predict_hotel_shift(hotel_name, shift):
    best_model = best_model_map[hotel_name][shift]
    config = HOTEL_CONFIG[hotel_name]
    prefix = config["model_prefix"]
    final_order = config["column_order"]
    lag_cols = config["lag_cols"]
    feats = {}
    # Build holiday flags from output data for today (if available)
    if "idx_today_output" in st.session_state and st.session_state.idx_today_output is not None:
        row_out = st.session_state.output_df.loc[st.session_state.idx_today_output]
        def outcol(c): return safe_int(row_out.get(c, None))
        feats["Ramadan_dummy"] = outcol("IsStartOfRamadhan") or outcol("IsMidRamadhan") or outcol("IsEndOfRamadhan")
        feats["Moharram_dummy"] = outcol("IsStartOfMoharam") or outcol("IsMidMoharam") or outcol("IsEndOfMoharam")
        feats["Eid_Fetr_dummy"] = outcol("IsFetr")
        feats["Norooz_dummy"] = outcol("IsNorooz")
        feats["Sizdah-be-Dar_dummy"] = outcol("Is13BeDar")
        eEarly = outcol("IsEarlyEsfand")
        eLate = outcol("IsLateEsfand")
        feats["Esfand_dummy"] = int(eEarly or eLate)
        feats["Last 5 Days of Esfand_dummy"] = outcol("IsLastDaysOfTheYear")
        feats["Hol_holiday"] = outcol("Hol_holiday")
        feats["Hol_none"] = outcol("Hol_none")
        feats["Hol_religious_holiday"] = outcol("Hol_religious_holiday")
        feats["Yalda_dummy"] = outcol("Yalda_dummy")
    else:
        for col in ["Ramadan_dummy","Moharram_dummy","Eid_Fetr_dummy","Norooz_dummy",
                    "Sizdah-be-Dar_dummy","Esfand_dummy","Last 5 Days of Esfand_dummy",
                    "Hol_holiday","Hol_none","Hol_religious_holiday","Yalda_dummy"]:
            feats[col] = 0
    # Weekday one-hot from today's date
    wd = datetime.datetime.now(tehran_tz).date().weekday()
    for i in range(7):
        feats[f"WD_{i}"] = 1 if i == wd else 0
    # Lag features from input data (using global idx_today_input)
    for i in range(1, 16):
        row_i = st.session_state.idx_today_input - i
        total = 0.0
        for c in lag_cols:
            try:
                total += float(st.session_state.input_df.loc[row_i, c])
            except:
                pass
        feats[f"Lag{i}_EmptyRooms"] = total
    row_vals = [feats.get(c, 0.0) for c in final_order]
    X_today = pd.DataFrame([row_vals], columns=final_order)
    model_path = f"results/{prefix}/{best_model}_{prefix}{shift}.pkl"
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        st.error(f"[Hotel {hotel_name}, shift={shift}] Model error: {e}")
        return np.nan
    if best_model in ["holt_winters", "exp_smoothing"]:
        return forecast_univariate_statsmodels(loaded_model, shift)
    elif best_model == "moving_avg":
        return forecast_moving_avg(loaded_model)
    elif best_model == "ts_decomp_reg":
        return forecast_ts_decomp_reg(loaded_model, X_today, shift)
    else:
        try:
            y_pred = loaded_model.predict(X_today)
            return float(y_pred[0]) if len(y_pred) > 0 else np.nan
        except Exception as e:
            st.error(f"Prediction error for {model_path}: {e}")
            return np.nan

def predict_chain_shift(shift):
    bestm = chain_shift_models[shift]
    chain_cfg = {
      "lag_cols": ["Blank"],
      "column_order": [
        "Ramadan_dummy","Ashoora_dummy","Eid_Fetr_dummy","Norooz_dummy",
        "Sizdah-be-Dar_dummy","Yalda_dummy","Last 5 Days of Esfand_dummy",
        "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms",
        "Lag5_EmptyRooms","Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms",
        "Lag9_EmptyRooms","Lag10_EmptyRooms",
        "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6",
        "Hol_holiday","Hol_none","Hol_religious_holiday"
      ]
    }
    feats = {}
    # Use holiday_map from output (if available)
    if "st_idx_today_output" in st.session_state and st.session_state.idx_today_output is not None:
        row_out = st.session_state.output_df.loc[st.session_state.idx_today_output]
        def outcol(c): return safe_int(row_out.get(c, None))
        for key in ["Ramadan_dummy","Ashoora_dummy","Eid_Fetr_dummy","Norooz_dummy",
                    "Sizdah-be-Dar_dummy","Yalda_dummy","Last 5 Days of Esfand_dummy",
                    "Hol_holiday","Hol_none","Hol_religious_holiday"]:
            feats[key] = outcol(key.replace("_dummy", ""))
    else:
        for key in ["Ramadan_dummy","Ashoora_dummy","Eid_Fetr_dummy","Norooz_dummy",
                    "Sizdah-be-Dar_dummy","Yalda_dummy","Last 5 Days of Esfand_dummy",
                    "Hol_holiday","Hol_none","Hol_religious_holiday"]:
            feats[key] = 0
    wd = datetime.datetime.now(tehran_tz).date().weekday()
    for i in range(7):
        feats[f"WD_{i}"] = 1 if i == wd else 0
    for i in range(1, 11):
        row_i = st.session_state.idx_today_input - i
        total = 0.0
        for c in chain_cfg["lag_cols"]:
            try:
                total += float(st.session_state.input_df.loc[row_i, c])
            except:
                pass
        feats[f"Lag{i}_EmptyRooms"] = total
    row_vals = [feats.get(c, 0.0) for c in chain_cfg["column_order"]]
    X_chain = pd.DataFrame([row_vals], columns=chain_cfg["column_order"])
    mp = f"results/Chain/{bestm}_Chain{shift}.pkl"
    try:
        with open(mp, "rb") as f:
            loaded_chain = pickle.load(f)
    except Exception as e:
        st.error(f"[Chain shift={shift}] Model error: {e}")
        return np.nan
    if bestm in ["holt_winters", "exp_smoothing"]:
        return forecast_univariate_statsmodels(loaded_chain, shift)
    elif bestm == "moving_avg":
        return forecast_moving_avg(loaded_chain)
    elif bestm == "ts_decomp_reg":
        return forecast_ts_decomp_reg(loaded_chain, X_chain, shift)
    else:
        try:
            ypred = loaded_chain.predict(X_chain)
            return float(ypred[0]) if len(ypred) > 0 else np.nan
        except Exception as e:
            st.error(f"Prediction error for chain model {mp}: {e}")
            return np.nan

def get_day_label(shift):
    if shift == 0:
        return "امروز"
    elif shift == 1:
        return "فردا"
    elif shift == 2:
        return "پسفردا"
    else:
        return "سه روز بعد"

##############################################################################
#  MAIN PAGE
##############################################################################
def main_page():
    load_css()
    st.image("tmoble.png", width=180)
    
    # LOGIN
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.markdown("<div class='login-box'><h2>ورود به داشبورد فروش اوپک</h2></div>", unsafe_allow_html=True)
        with st.container():
            with st.columns(3)[1]:
                password = st.text_input("", type="password", label_visibility="collapsed", placeholder="رمز عبور را وارد کنید")
                if st.button("ورود", key="login_button"):
                    if password == "1234":
                        st.session_state.logged_in = True
                    else:
                        st.error("رمز عبور اشتباه است!")
        return

    # Load data
    input_df, output_df = read_main_dfs()
    if input_df.empty or output_df.empty:
        st.error("داده‌ها کامل نیستند.")
        return
    system_today = datetime.datetime.now(tehran_tz).date()
    st.session_state.input_df = input_df
    st.session_state.output_df = output_df
    matches = input_df.index[input_df["parsed_input_date"] == system_today].tolist()
    if not matches:
        st.warning("برای تاریخ امروز سطری در ورودی یافت نشد.")
        return
    st.session_state.idx_today_input = matches[0]
    idx_today_input = st.session_state.idx_today_input
    matches_out = output_df.index[output_df["parsed_output_date"] == system_today].tolist()
    idx_today_output = matches_out[0] if matches_out else None
    st.session_state.idx_today_output = idx_today_output

    jalali_today = jdatetime.date.fromgregorian(date=system_today)
    greg_str = system_today.strftime("%Y/%m/%d")
    jalali_str = jalali_today.strftime("%Y/%m/%d")
    st.markdown(f'<div class="scoreboard">تاریخ میلادی: {greg_str} &nbsp;&nbsp;|&nbsp;&nbsp; تاریخ جلالی: {jalali_str}</div>', unsafe_allow_html=True)
    
    # Toggle for prediction view
    prediction_view_option = st.radio("سناریو را انتخاب کنید:", 
                                      ["خوش‌بینانه", "واقع‌بینانه", "بدبینانه"],
                                      index=1, horizontal=True)
    
    day_results = []
    hotels_list = list(best_model_map.keys())
    for shift in range(4):
        # Hotel predictions & chain prediction
        hotel_preds = {h: predict_hotel_shift(h, shift) for h in hotels_list}
        sum_houses = sum(v for v in hotel_preds.values() if not pd.isna(v))
        chain_pred = predict_chain_shift(shift)
        row_future = idx_today_input + shift
        try:
            future_blank = float(input_df.loc[row_future, "Blank"])
        except:
            future_blank = 0.0
        try:
            uncertain_val = float(input_df.loc[row_future, "Hold"])
        except:
            uncertain_val = 0.0
        try:
            week_day = input_df.loc[row_future, "Week Day"]
        except:
            week_day = "-"
        if chain_pred is None or np.isnan(chain_pred):
            chain_pred = 0.0
        if future_blank is None or np.isnan(future_blank):
            future_blank = 0.0
        whole_chain = min(chain_pred, future_blank)
        robust = 0.5 * (sum_houses + whole_chain)
        pn = int(round(min(robust, future_blank) - uncertain_val))
        cur = int(round(future_blank))
        pf_tel = int(round(sum_houses))
        pf_kli = int(round(whole_chain))
        happy = int(round(min(cur, min(pf_tel, pf_kli, pn))))
        sad = int(round(min(cur, max(pf_tel, pf_kli, pn))))
        realistic = int(round(min(cur, round((pf_tel + pf_kli + pn)/3))))
        day_results.append({
            "shift": shift,
            "label": get_day_label(shift),
            "روز هفته": week_day,
            "تعداد خالی فعلی": cur,
            "غیرقطعی": int(uncertain_val),
            "پیش‌بینی پیشخور تلفیقی": pf_tel,
            "پیش‌بینی پیشخور کلی": pf_kli,
            "پیش‌بینی نمایشی": pn,
            "پیش‌بینی نهایی خوشبینانه": happy,
            "پیش‌بینی نهایی بدبینانه": sad,
            "پیش‌بینی نهایی واقع‌بینانه": realistic,
            "hotel_preds": hotel_preds
        })
    
    st.subheader("فروش اوپک پیشنهادی")
    cols = st.columns(4)
    # For each day, choose the prediction type based on toggle and subtract 10.
    for idx, (col, row) in enumerate(zip(cols, day_results)):
        if prediction_view_option == "خوش‌بینانه":
            base_pred = row["پیش‌بینی نهایی خوشبینانه"]
        elif prediction_view_option == "بدبینانه":
            base_pred = row["پیش‌بینی نهایی بدبینانه"]
        else:
            base_pred = row["پیش‌بینی نهایی واقع‌بینانه"]
        final_value = max(base_pred - 10, 0)
        extra_content = f"""
        <div id="pred-extra-{idx}" class="extra-text">
          <div>تعداد خالی فعلی: {row['تعداد خالی فعلی']}</div>
        </div>
        """
        html_code = f"""
        <html>
        <head>
          <style>
            .score-box {{
              background: linear-gradient(135deg, #FFFFFF, #F0F0F0);
              color: #333;
              cursor: pointer;
              padding: 20px;
              border-radius: 5px;
              text-align: center;
              width: 100%;
              box-sizing: border-box;
            }}
            .extra-text {{
              display: none;
              margin-top: 10px;
              font-size: 14px;
              font-family: "Tahoma", sans-serif !important;
            }}
          </style>
          <script>
            function togglePredExtra_{idx}() {{
              var x = document.getElementById("pred-extra-{idx}");
              if (x.style.display === "none" || x.style.display === "") {{
                x.style.display = "block";
              }} else {{
                x.style.display = "none";
              }}
            }}
          </script>
        </head>
        <body>
          <div class="score-box" onclick="togglePredExtra_{idx}()">
            <div><b>{row['label']}</b></div>
            <div><b>فروش اوپک پیشنهادی: {final_value}</b></div>
            {extra_content}
          </div>
        </body>
        </html>
        """
        with col:
            components.html(html_code, height=150, width=200)
    
    # Display critical groups for each day
    st.write("---")
    for day in day_results:
        hotel_preds_for_day = day.get("hotel_preds", {})
        filtered_hotels = [(h, val) for (h, val) in hotel_preds_for_day.items() if (not pd.isna(val)) and (val > 3)]
        if filtered_hotels:
            total_empties = sum(val for _, val in filtered_hotels)
            if total_empties > 0:
                filtered_hotels.sort(key=lambda x: x[1], reverse=True)
                cutoff = 0.8 * total_empties
                cumsum = 0.0
                critical_hotels = []
                for h, empties in filtered_hotels:
                    cumsum += empties
                    critical_hotels.append(h)
                    if cumsum >= cutoff:
                        break
                if critical_hotels:
                    hotels_str = " - ".join([hotel_name_map.get(h, h) for h in critical_hotels])
                    st.info(f"مجموعه‌های پیشنهادی برای فروش اوپک {day['label']}: {hotels_str}")

def main():
    st.set_page_config(page_title="داشبورد فروش اوپک", page_icon="📈", layout="wide")
    main_page()

if __name__ == "__main__":
    main()
