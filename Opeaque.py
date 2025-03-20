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
#                   HELPER FUNCTIONS: GOOGLE SHEETS, DATE PARSING, ETC.
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
    except Exception:
        return None

def parse_output_date_str(date_str):
    s = str(date_str).strip()
    try:
        dt = datetime.datetime.strptime(s, "%A, %B %d, %Y at %I:%M:%S %p")
        return dt.date()
    except Exception:
        return None

def safe_int(val):
    if val is None:
        return 0
    return 1 if str(val).strip() == "1" else 0

def compute_avg_for_weekday(input_df, target_weekday, days_interval):
    system_today = datetime.datetime.now(tehran_tz).date()
    start_date = system_today - datetime.timedelta(days=days_interval)
    mask = (
        (input_df["parsed_input_date"] >= start_date) &
        (input_df["parsed_input_date"] <= system_today) &
        (input_df["parsed_input_date"].apply(lambda d: d.weekday() if pd.notnull(d) else -1) == target_weekday)
    )
    filtered = input_df[mask]
    if not filtered.empty:
        try:
            return filtered["Blank"].astype(float).mean()
        except Exception:
            return 0
    return 0

##############################################################################
#           FORECAST HELPERS: UNIVARIATE, MOVING AVG, TS DECOMP REG
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
    except Exception:
        resid_pred = 0.0
    return last_trend + seas_vals[pos] + resid_pred

##############################################################################
#                          CUSTOM CSS & FONT SETUP
##############################################################################

def load_css():
    st.markdown(
        """
        <style>
        * {
            font-family: "Tahoma", sans-serif !important;
        }
        body {
            background-color: #eef2f7;
            direction: rtl;
            text-align: center;
        }
        .header {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 28px;
            text-align: center;
        }
        .scoreboard {
            background-color: #ecf0f1;
            border: 2px solid #34495e;
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 8px;
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: #34495e;
        }
        table, th, td {
            text-align: center !important;
        }
        .stTable {
            font-size: 18px;
        }
        .login-box {
            max-width: 300px;
            margin: auto;
            padding: 40px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

##############################################################################
#                           HOTEL NAME MAPPING
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

##############################################################################
#               پیش‌بینی پیشخور برای هتل و زنجیره (همان منطق اصلی)
##############################################################################

def pishkhor_for_hotel(hotel_name, start_date, input_df, output_df, best_model_map, HOTEL_CONFIG):
    model_tag = best_model_map[hotel_name][0]
    config = HOTEL_CONFIG[hotel_name]
    prefix = config["model_prefix"]
    final_order = config["column_order"]
    lag_cols = config["lag_cols"]
    model_path = f"results/{prefix}/{model_tag}_{prefix}0.pkl"
    predicted_cache = {}

    def build_shift0_features(target_date):
        row_match = output_df.index[output_df["parsed_output_date"] == target_date].tolist()
        holiday_feats = {}
        if row_match:
            row_out = output_df.loc[row_match[0]]
            def outcol(c):
                return safe_int(row_out.get(c, None))
            holiday_feats["Ramadan_dummy"] = outcol("IsStartOfRamadhan") or outcol("IsMidRamadhan") or outcol("IsEndOfRamadhan")
            holiday_feats["Moharram_dummy"] = outcol("IsStartOfMoharam") or outcol("IsMidMoharam") or outcol("IsEndOfMoharam")
            holiday_feats["Ashoora_dummy"]  = outcol("IsTasooaAshoora")
            holiday_feats["Arbain_dummy"]   = outcol("IsArbain")
            holiday_feats["Eid_Fetr_dummy"] = outcol("IsFetr")
            holiday_feats["Shabe_Ghadr_dummy"] = outcol("IsShabeGhadr")
            holiday_feats["Sizdah-be-Dar_dummy"] = outcol("Is13BeDar")
            eEarly = outcol("IsEarlyEsfand")
            eLate  = outcol("IsLateEsfand")
            holiday_feats["Esfand_dummy"] = int(eEarly or eLate)
            holiday_feats["Last 5 Days of Esfand_dummy"] = outcol("IsLastDaysOfTheYear")
            holiday_feats["Norooz_dummy"] = outcol("IsNorooz")
            holiday_feats["Hol_holiday"]  = outcol("Hol_holiday")
            holiday_feats["Hol_none"]     = outcol("Hol_none")
            holiday_feats["Hol_religious_holiday"] = outcol("Hol_religious_holiday")
            holiday_feats["Yalda_dummy"]  = outcol("Yalda_dummy")
        else:
            for fcol in ["Ramadan_dummy","Moharram_dummy","Ashoora_dummy","Arbain_dummy","Eid_Fetr_dummy","Shabe_Ghadr_dummy",
                         "Sizdah-be-Dar_dummy","Esfand_dummy","Last 5 Days of Esfand_dummy","Norooz_dummy",
                         "Hol_holiday","Hol_none","Hol_religious_holiday","Yalda_dummy"]:
                holiday_feats[fcol] = 0

        wd = target_date.weekday()
        for i in range(7):
            holiday_feats[f"WD_{i}"] = 1 if (i == wd) else 0

        def get_empties_for_date(d_):
            if d_ in predicted_cache:
                return predicted_cache[d_]
            row_m = input_df.index[input_df["parsed_input_date"] == d_].tolist()
            if not row_m:
                return 0.0
            ridx = row_m[0]
            total_ = 0.0
            for c in lag_cols:
                try:
                    total_ += float(input_df.loc[ridx, c])
                except:
                    pass
            return total_

        for lag in range(1, 16):
            dlag = target_date - datetime.timedelta(days=lag)
            holiday_feats[f"Lag{lag}_EmptyRooms"] = get_empties_for_date(dlag)

        row_vals = [holiday_feats.get(col, 0.0) for col in final_order]
        return pd.DataFrame([row_vals], columns=final_order)

    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
    except:
        return [np.nan]*4

    results_4 = []
    for i in range(4):
        d_ = start_date + datetime.timedelta(days=i)
        feats_df = build_shift0_features(d_)
        if model_tag in ["holt_winters", "exp_smoothing"]:
            pred_val = forecast_univariate_statsmodels(loaded_model, 0)
        elif model_tag == "moving_avg":
            pred_val = forecast_moving_avg(loaded_model)
        elif model_tag == "ts_decomp_reg":
            pred_val = forecast_ts_decomp_reg(loaded_model, feats_df, 0)
        else:
            try:
                pp = loaded_model.predict(feats_df)
                pred_val = float(pp[0]) if len(pp) > 0 else np.nan
            except:
                pred_val = np.nan
        if pred_val >= 100:
            pred_val = 100
        results_4.append(pred_val)
        predicted_cache[d_] = pred_val

    return results_4

def pishkhor_for_chain(start_date, input_df, output_df, chain_shift_models):
    bestm0 = chain_shift_models[0]
    mp = f"results/Chain/{bestm0}_Chain0.pkl"
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
    try:
        with open(mp, "rb") as f:
            loaded_chain = pickle.load(f)
    except:
        return [np.nan]*4

    predicted_cache = {}

    def build_chain0_features(tdate):
        feats = {}
        row_match = output_df.index[output_df["parsed_output_date"] == tdate].tolist()
        if row_match:
            row_out = output_df.loc[row_match[0]]
            def outcol(c):
                return safe_int(row_out.get(c, None))
            feats["Ramadan_dummy"] = outcol("IsStartOfRamadhan") or outcol("IsMidRamadhan") or outcol("IsEndOfRamadhan")
            feats["Ashoora_dummy"] = outcol("IsTasooaAshoora")
            feats["Eid_Fetr_dummy"] = outcol("IsFetr")
            feats["Norooz_dummy"]  = outcol("IsNorooz")
            feats["Sizdah-be-Dar_dummy"] = outcol("Is13BeDar")
            feats["Yalda_dummy"]   = outcol("Yalda_dummy")
            feats["Last 5 Days of Esfand_dummy"] = outcol("IsLastDaysOfTheYear")
            feats["Hol_holiday"]   = outcol("Hol_holiday")
            feats["Hol_none"]      = outcol("Hol_none")
            feats["Hol_religious_holiday"] = outcol("Hol_religious_holiday")
        else:
            for c_ in ["Ramadan_dummy","Ashoora_dummy","Eid_Fetr_dummy","Norooz_dummy",
                       "Sizdah-be-Dar_dummy","Yalda_dummy","Last 5 Days of Esfand_dummy",
                       "Hol_holiday","Hol_none","Hol_religious_holiday"]:
                feats[c_] = 0

        wd = tdate.weekday()
        for i in range(7):
            feats[f"WD_{i}"] = 1 if (i == wd) else 0

        def get_blank_for_date(dt_):
            if dt_ in predicted_cache:
                return predicted_cache[dt_]
            row_m = input_df.index[input_df["parsed_input_date"] == dt_].tolist()
            if not row_m:
                return 0.0
            try:
                return float(input_df.loc[row_m[0], "Blank"])
            except:
                return 0.0

        for i in range(1, 11):
            dlag = tdate - datetime.timedelta(days=i)
            feats[f"Lag{i}_EmptyRooms"] = get_blank_for_date(dlag)

        row_vals = [feats.get(c, 0.0) for c in chain_cfg["column_order"]]
        return pd.DataFrame([row_vals], columns=chain_cfg["column_order"])

    results_4 = []
    for i in range(4):
        d_ = start_date + datetime.timedelta(days=i)
        X_chain = build_chain0_features(d_)
        if bestm0 in ["holt_winters", "exp_smoothing"]:
            val = forecast_univariate_statsmodels(loaded_chain, 0)
        elif bestm0 == "moving_avg":
            val = forecast_moving_avg(loaded_chain)
        elif bestm0 == "ts_decomp_reg":
            val = forecast_ts_decomp_reg(loaded_chain, X_chain, 0)
        else:
            try:
                pred_ = loaded_chain.predict(X_chain)
                val = float(pred_[0]) if len(pred_) > 0 else np.nan
            except:
                val = np.nan

        results_4.append(val)
        predicted_cache[d_] = val

    return results_4

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
#                       READ MAIN DATA (CACHED)
##############################################################################

@st.cache_data
def read_main_dfs():
    service = create_gsheets_connection()
    SPREADSHEET_ID = "1LI0orqvqci1d75imMfHKxZ512rUUlpA7P1ZYjV-uVO0"

    # Input data
    input_df = read_sheet_values(service, SPREADSHEET_ID, "Input", "A1:ZZ10000")
    input_df["Date"] = input_df.iloc[:, 3]
    input_df["Blank"] = input_df.iloc[:, 2]
    input_df["Hold"] = input_df.iloc[:, 1]
    input_df["Week Day"] = input_df.iloc[:, 4]
    input_df["parsed_input_date"] = input_df["Date"].apply(parse_input_date_str)

    # Output data
    output_df = read_sheet_values(service, SPREADSHEET_ID, "Output", "A1:ZZ10000")
    output_df["parsed_output_date"] = output_df["Date"].apply(parse_output_date_str)

    return input_df, output_df

##############################################################################
#                           MAIN DASHBOARD PAGE
##############################################################################

def main_page():
    load_css()
    
    # ---------- LOGIN SECTION ----------
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

    # ---------- MAIN DASHBOARD ----------
    st.image("tmoble.png", width=180)
    st.markdown('<div class="header">داشبورد فروش اوپک</div>', unsafe_allow_html=True)
    system_today = datetime.datetime.now(tehran_tz).date()
    jalali_today = jdatetime.date.fromgregorian(date=system_today)
    greg_str = system_today.strftime("%Y/%m/%d")
    jalali_str = jalali_today.strftime("%Y/%m/%d")
    st.markdown(
        f'<div class="scoreboard">تاریخ میلادی: {greg_str} &nbsp;&nbsp;|&nbsp;&nbsp; تاریخ جلالی: {jalali_str}</div>',
        unsafe_allow_html=True
    )
    
    input_df, output_df = read_main_dfs()
    if input_df.empty:
        st.error("ورودی خالی است.")
        return
    if output_df.empty:
        st.error("خروجی خالی است.")
        return

    matches = input_df.index[input_df["parsed_input_date"] == system_today].tolist()
    if not matches:
        st.warning("برای تاریخ امروز سطری در ورودی یافت نشد.")
        return
    idx_today_input = matches[0]
    
    try:
        future_blank = float(input_df.loc[idx_today_input, "Blank"])
    except:
        future_blank = 0.0

    try:
        uncertain_val = float(input_df.loc[idx_today_input, "Hold"])
    except:
        uncertain_val = 0.0

    # ------------------- MODEL CONFIGURATIONS -------------------
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
    chain_shift_models = ["linear_reg","xgboost","xgboost","xgboost","linear_reg","xgboost","linear_reg"]
    HOTEL_CONFIG = {
       "Ashrafi": {
         "model_prefix": "Ashrafi",
         "lag_cols": ["AshrafiN", "AshrafiS"],
         "column_order": [
            "Ramadan_dummy","Moharram_dummy","Eshoora_dummy","Eid_Fetr_dummy","Norooz_dummy","Sizdah-be-Dar_dummy",
            "Lag1_EmptyRooms","Lag2_EmptyRooms","Lag3_EmptyRooms","Lag4_EmptyRooms","Lag5_EmptyRooms",
            "Lag6_EmptyRooms","Lag7_EmptyRooms","Lag8_EmptyRooms","Lag9_EmptyRooms","Lag10_EmptyRooms",
            "WD_0","WD_1","WD_2","WD_3","WD_4","WD_5","WD_6"
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

    # ------------------- HOLIDAY FLAGS -------------------
    idx_today_output = None
    matches_out = output_df.index[output_df["parsed_output_date"] == system_today].tolist()
    if matches_out:
        idx_today_output = matches_out[0]
    if idx_today_output is not None:
        row_output_today = output_df.loc[idx_today_output]
        def outcol(c):
            return safe_int(row_output_today.get(c, None))
        Ramadan = outcol("IsStartOfRamadhan") or outcol("IsMidRamadhan") or outcol("IsEndOfRamadhan")
        Moharram = outcol("IsStartOfMoharam") or outcol("IsMidMoharam") or outcol("IsEndOfMoharam")
        Ashoora = outcol("IsTasooaAshoora")
        Arbain  = outcol("IsArbain")
        Fetr    = outcol("IsFetr")
        Shabe   = outcol("IsShabeGhadr")
        S13     = outcol("Is13BeDar")
        eEarly  = outcol("IsEarlyEsfand")
        eLate   = outcol("IsLateEsfand")
        Esfand  = int(eEarly or eLate)
        L5      = outcol("IsLastDaysOfTheYear")
        Nrz     = outcol("IsNorooz")
        HolHol  = outcol("Hol_holiday")
        HolNone = outcol("Hol_none")
        HolRel  = outcol("Hol_religious_holiday")
        Yalda   = outcol("Yalda_dummy")
    else:
        Ramadan = Moharram = Ashoora = Arbain = Fetr = Shabe = S13 = 0
        Esfand = L5 = Nrz = HolHol = HolNone = HolRel = Yalda = 0

    dow = system_today.weekday()
    WD_ = {f"WD_{i}": 1 if i == dow else 0 for i in range(7)}

    def sum_cols_for_row(irow, colnames):
        if irow < 0 or irow >= len(input_df):
            return 0.0
        total = 0.0
        for c in colnames:
            try:
                total += float(input_df.loc[irow, c])
            except:
                pass
        return total

    def predict_hotel_shift(hotel_name, shift):
        best_model = best_model_map[hotel_name][shift]
        config = HOTEL_CONFIG[hotel_name]
        prefix = config["model_prefix"]
        final_order = config["column_order"]
        lag_cols = config["lag_cols"]

        feats = {}
        if idx_today_output is not None:
            row_output_today = output_df.loc[idx_today_output]
            def outcol(c):
                return safe_int(row_output_today.get(c, None))
            feats["Ramadan_dummy"] = outcol("IsStartOfRamadhan") or outcol("IsMidRamadhan") or outcol("IsEndOfRamadhan")
            feats["Moharram_dummy"] = outcol("IsStartOfMoharam") or outcol("IsMidMoharam") or outcol("IsEndOfMoharam")
            feats["Ashoora_dummy"] = outcol("IsTasooaAshoora")
            feats["Arbain_dummy"] = outcol("IsArbain")
            feats["Eid_Fetr_dummy"] = outcol("IsFetr")
            feats["Shabe_Ghadr_dummy"] = outcol("IsShabeGhadr")
            feats["Sizdah-be-Dar_dummy"] = outcol("Is13BeDar")
            eEarly = outcol("IsEarlyEsfand")
            eLate = outcol("IsLateEsfand")
            feats["Esfand_dummy"] = int(eEarly or eLate)
            feats["Last 5 Days of Esfand_dummy"] = outcol("IsLastDaysOfTheYear")
            feats["Norooz_dummy"] = outcol("IsNorooz")
            feats["Hol_holiday"] = outcol("Hol_holiday")
            feats["Hol_none"] = outcol("Hol_none")
            feats["Hol_religious_holiday"] = outcol("Hol_religious_holiday")
            feats["Yalda_dummy"] = outcol("Yalda_dummy")
        else:
            for c_ in ["Ramadan_dummy","Moharram_dummy","Ashoora_dummy","Arbain_dummy","Eid_Fetr_dummy","Shabe_Ghadr_dummy",
                       "Sizdah-be-Dar_dummy","Esfand_dummy","Last 5 Days of Esfand_dummy","Norooz_dummy",
                       "Hol_holiday","Hol_none","Hol_religious_holiday","Yalda_dummy"]:
                feats[c_] = 0

        for i in range(7):
            feats[f"WD_{i}"] = WD_.get(f"WD_{i}", 0)

        for i in range(1, 16):
            row_i = idx_today_input - i
            feats[f"Lag{i}_EmptyRooms"] = sum_cols_for_row(row_i, lag_cols)

        row_vals = [feats.get(c, 0.0) for c in final_order]
        X_today = pd.DataFrame([row_vals], columns=final_order)
        model_path = f"results/{prefix}/{best_model}_{prefix}{shift}.pkl"
        try:
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
        except FileNotFoundError as e:
            st.error(f"[Hotel {hotel_name}, shift={shift}] Model file not found: {model_path}")
            return np.nan
        except Exception as e:
            st.error(f"[Hotel {hotel_name}, shift={shift}] Error loading model {model_path}: {e}")
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
        if idx_today_output is not None:
            row_out = output_df.loc[idx_today_output]
            def outcol(c):
                return safe_int(row_out.get(c, None))
            feats["Ramadan_dummy"] = outcol("IsStartOfRamadhan") or outcol("IsMidRamadhan") or outcol("IsEndOfRamadhan")
            feats["Ashoora_dummy"] = outcol("IsTasooaAshoora")
            feats["Eid_Fetr_dummy"] = outcol("IsFetr")
            feats["Norooz_dummy"]  = outcol("IsNorooz")
            feats["Sizdah-be-Dar_dummy"] = outcol("Is13BeDar")
            feats["Yalda_dummy"]   = outcol("Yalda_dummy")
            feats["Last 5 Days of Esfand_dummy"] = outcol("IsLastDaysOfTheYear")
            feats["Hol_holiday"]   = outcol("Hol_holiday")
            feats["Hol_none"]      = outcol("Hol_none")
            feats["Hol_religious_holiday"] = outcol("Hol_religious_holiday")
        else:
            for c_ in ["Ramadan_dummy","Ashoora_dummy","Eid_Fetr_dummy","Norooz_dummy",
                       "Sizdah-be-Dar_dummy","Yalda_dummy","Last 5 Days of Esfand_dummy",
                       "Hol_holiday","Hol_none","Hol_religious_holiday"]:
                feats[c_] = 0

        for i in range(7):
            feats[f"WD_{i}"] = WD_.get(f"WD_{i}", 0)

        for i in range(1, 11):
            row_i = idx_today_input - i
            feats[f"Lag{i}_EmptyRooms"] = sum_cols_for_row(row_i, chain_cfg["lag_cols"])
        row_vals = [feats.get(c, 0.0) for c in chain_cfg["column_order"]]
        X_chain = pd.DataFrame([row_vals], columns=chain_cfg["column_order"])
        mp = f"results/Chain/{bestm}_Chain{shift}.pkl"
        
        try:
            with open(mp, "rb") as f:
                loaded_chain = pickle.load(f)
        except FileNotFoundError as e:
            st.error(f"[Chain shift={shift}] Model file not found: {mp}")
            return np.nan
        except Exception as e:
            st.error(f"[Chain shift={shift}] Error loading model {mp}: {e}")
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

    # ------------------- CALCULATE DAY RESULTS -------------------
    day_results = []
    for shift in range(4):
        hotels_list = list(best_model_map.keys())
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
        try:
            pn = int(round(min(robust, future_blank) - uncertain_val))
        except:
            pn = 0
        day_results.append({
            "shift": shift,
            "label": get_day_label(shift),
            "روز هفته": week_day,
            "تعداد خالی فعلی": int(round(future_blank)),
            "غیرقطعی": int(uncertain_val),
            "پیش بینی نمایشی": pn,
            "hotel_preds": hotel_preds
        })
    
    pishkhor_hotels_dict = {}
    for h_ in best_model_map.keys():
        p4 = pishkhor_for_hotel(h_, system_today, input_df, output_df, best_model_map, HOTEL_CONFIG)
        pishkhor_hotels_dict[h_] = p4
    
    pishkhor_telefiqi = []
    for i in range(4):
        s_ = 0.0
        for h_ in best_model_map.keys():
            val_ = pishkhor_hotels_dict[h_][i]
            if not pd.isna(val_):
                s_ += val_
        pishkhor_telefiqi.append(s_)
    
    pishkhor_chain_vals = pishkhor_for_chain(system_today, input_df, output_df, chain_shift_models)
    
    for i in range(4):
        day_results[i]["پیش‌بینی پیشخور تلفیقی"] = int(round(pishkhor_telefiqi[i]))
        day_results[i]["پیش‌بینی پیشخور کلی"]   = int(round(pishkhor_chain_vals[i]))
    
        pf_tel = day_results[i]["پیش‌بینی پیشخور تلفیقی"]
        pf_kli = day_results[i]["پیش‌بینی پیشخور کلی"]
        pn     = day_results[i]["پیش بینی نمایشی"]
        cur    = day_results[i]["تعداد خالی فعلی"]
    
        day_results[i]["پیش‌بینی نهایی خوشبینانه"] = int(round(min(cur, min(pf_tel, pf_kli, pn))))
        day_results[i]["پیش‌بینی نهایی بدبینانه"] = int(round(min(cur, max(pf_tel, pf_kli, pn))))
        avg_val = (pf_tel + pf_kli + pn) / 3
        day_results[i]["پیش‌بینی نهایی واقع‌بینانه"] = int(round(min(cur, round(avg_val))))
    
    # ------------------- DISPLAY CARDS -------------------
    # For each day, show a card with the value فروش اوپک پیشنهادی which is
    # (پیش‌بینی نهایی بدبینانه minus 10) clamped to a minimum of 0.
    K = 10
    st.subheader("فروش اوپک پیشنهادی")
    cols = st.columns(4)
    for idx, (col, row) in enumerate(zip(cols, day_results)):
        final_bad = max(row["پیش‌بینی نهایی بدبینانه"] - K, 0)
        extra_content = f"""
        <div id="pred-extra-{idx}" class="extra-text">
        <div>تعداد خالی فعلی: {row['تعداد خالی فعلی']}</div>
        <div>غیرقطعی: {row['غیرقطعی']}</div>
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
            <div><b>فروش اوپک پیشنهادی: {final_bad}</b></div>
            {extra_content}
        </div>
        </body>
        </html>
        """
        with col:
            components.html(html_code, height=150, width=200)
    
    # ------------------- DISPLAY CRITICAL GROUPS -------------------
    st.write("---")
    st.subheader("مجموعه‌های پیشنهادی برای فروش اوپک امروز")
    today_result = next(item for item in day_results if item["shift"] == 0)
    hotel_preds_for_today = today_result.get("hotel_preds", {})
    filtered_hotels = [(h, val) for (h, val) in hotel_preds_for_today.items() if (not pd.isna(val)) and (val > 3)]
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
                st.info(f"مجموعه‌های پیشنهادی برای فروش اوپک امروز:\n{hotels_str}")

def main():
    st.set_page_config(page_title="داشبورد فروش اوپک", page_icon="📈", layout="wide")
    main_page()

if __name__ == "__main__":
    main()
