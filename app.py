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
import zoneinfo  # Available in Python 3.9+
tehran_tz = zoneinfo.ZoneInfo("Asia/Tehran")

##############################################################################
#                   HELPER FUNCTIONS: GOOGLE SHEETS, DATE PARSING, ETC.
##############################################################################

@st.cache_data
def create_gsheets_connection():
    """Create a cached connection to Google Sheets (read-only) using Streamlit secrets."""
    service_account_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build('sheets', 'v4', credentials=creds)
    return service

def get_pickup_value_for_day(pivot_df, arrival_date, offset):
    """
    Returns the number of reservations (pickup count) for a given arrival_date and offset.
    For example, for offset=4, it returns the count of reservations where
    'تاریخ ورود میلادی' equals arrival_date and 'تاریخ معامله میلادی'
    equals arrival_date minus 4 days.
    """
    if arrival_date in pivot_df.index:
        try:
            return int(pivot_df.loc[arrival_date, f"pickup{offset}"])
        except:
            return 0
    return 0

def compute_avg_for_weekday(input_df, target_weekday, days_interval):
    """Compute average Blank for a given weekday over a given past period."""
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
        except:
            return 0
    return 0

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
    """Parses a date in 'YYYY/MM/DD' format from the Input sheet."""
    s = str(date_str).strip()
    try:
        dt = datetime.datetime.strptime(s, "%Y/%m/%d")
        return dt.date()
    except:
        return None

def parse_output_date_str(date_str):
    """Parses a date in 'Saturday, March 8, 2025 at 12:00:00 AM' format from Output sheet."""
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
    except:
        resid_pred = 0.0
    return last_trend + seas_vals[pos] + resid_pred

##############################################################################
#                            CUSTOM CSS & FONT SETUP
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
        </style>
        """,
        unsafe_allow_html=True
    )

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
#               PICKUP MODEL HELPERS (for the "مدل پیکآپ" column)
##############################################################################

import gspread

def convert_farsi_number(num):
    try:
        s = str(num).strip()
        if s == "" or s.lower() in ["nan", "none"]:
            return 1
        farsi_to_english = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
        converted = s.translate(farsi_to_english)
        return int(converted)
    except:
        return 1

@st.cache_data
def get_data_from_pickup_sheet():
    """Retrieve data from a Google Sheet (read-only) using credentials from Streamlit secrets."""
    scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    service_account_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        service_account_info, 
        scopes=scopes
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key("1D5ROCnoTKCFBQ8me8wLIri8mlaOUF4v1hsyC7LXIvAE").worksheet("Sheet1")
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    return df

def build_pickup_pivot(df):
    df = df[["تاریخ معامله میلادی", "تاریخ ورود میلادی", "تعداد شب"]].copy()
    df["تاریخ معامله میلادی"] = pd.to_datetime(df["تاریخ معامله میلادی"], format="%Y/%m/%d", errors="coerce")
    df["تاریخ ورود میلادی"] = pd.to_datetime(df["تاریخ ورود میلادی"], format="%Y/%m/%d", errors="coerce")
    df["تعداد شب"] = df["تعداد شب"].fillna(1)
    df["تعداد شب"] = df["تعداد شب"].apply(lambda x: convert_farsi_number(x))
    
    pivot_list = []
    unique_arrivals = df["تاریخ ورود میلادی"].dropna().dt.date.unique()
    
    for arrival in unique_arrivals:
        arrival_date = arrival
        row = {"تاریخ ورود میلادی": arrival_date}
        for offset in range(0, 11):
            target_deal_date = arrival_date - datetime.timedelta(days=offset)
            sub = df[
                (df["تاریخ ورود میلادی"].dt.date == arrival_date)
                & (df["تاریخ معامله میلادی"].dt.date == target_deal_date)
            ]
            row[f"pickup{offset}"] = len(sub)
            row[f"pickup_night{offset}"] = sub["تعداد شب"].sum()
        pivot_list.append(row)
    
    pivot_df = pd.DataFrame(pivot_list)
    pivot_df = pivot_df.set_index("تاریخ ورود میلادی").fillna(0)
    
    cols = []
    for offset in range(0, 11):
        cols.append(f"pickup{offset}")
        cols.append(f"pickup_night{offset}")
    pivot_df = pivot_df[cols]
    return pivot_df

def load_pickup_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

def predict_pickup_for_shift(arrival_date, pivot_df, shift):
    if arrival_date in pivot_df.index:
        feature_row = pivot_df.loc[arrival_date]
    else:
        feature_row = pd.Series({col: 0 for col in pivot_df.columns})

    X_features = feature_row.values.reshape(1, -1)
    
    if shift == 0:
        model_filename = "Pickup/linear_regression_model.pkl"
    else:
        model_filename = f"Pickup/linear_regression_model_shift_{shift}.pkl"
    
    try:
        model = load_pickup_model(model_filename)
    except FileNotFoundError as e:
        st.error(f"[Pickup] Model file not found: {model_filename}")
        return None
    except Exception as e:
        st.error(f"[Pickup] Error loading model {model_filename}: {e}")
        return None
    
    predicted_empty = model.predict(X_features)[0]
    return predicted_empty

##############################################################################
#                       READ MAIN DATA (CACHED)
##############################################################################

@st.cache_data
def read_main_dfs():
    """
    We read Input from 'Input' sheet for numeric columns (Blank, Hold, etc.),
    and Output from 'Output' sheet for holiday flags like IsStartOfRamadhan, etc.
    """
    service = create_gsheets_connection()
    SPREADSHEET_ID = "1LI0orqvqci1d75imMfHKxZ512rUUlpA7P1ZYjV-uVO0"

    # Input data
    input_df = read_sheet_values(service, SPREADSHEET_ID, "Input", "A1:ZZ10000")
    input_df["Date"] = input_df.iloc[:, 3]   # column D
    input_df["Blank"] = input_df.iloc[:, 2]  # column C
    input_df["parsed_input_date"] = input_df["Date"].apply(parse_input_date_str)

    # Output data
    output_df = read_sheet_values(service, SPREADSHEET_ID, "Output", "A1:ZZ10000")
    output_df["parsed_output_date"] = output_df["Date"].apply(parse_output_date_str)

    return input_df, output_df

##############################################################################
#                          FUZZY COLOR UTILS
##############################################################################

def fuzz_color(value, total=330):
    occ = (total - value) / total
    if occ >= 0.99:
        return 0  # Blue
    elif occ >= 0.96:
        return 1  # Green
    elif occ >= 0.93:
        return 2  # Yellow
    elif occ >= 0.90:
        return 3  # Red
    else:
        return 4  # Black

def union_fuzzy(colors):
    if not colors:
        return 0
    avg_val = sum(colors) / len(colors)
    return int(round(avg_val))

def color_code_to_hex(c):
    if c == 0:
        return "#4A90E2"
    elif c == 1:
        return "#7ED321"
    elif c == 2:
        return "#F5A623"
    elif c == 3:
        return "#D0021B"
    else:
        return "#333333"

##############################################################################
#                NEW: ADDITIONAL HELPERS FOR "پیش‌بینی پیشخور"
##############################################################################

def pishkhor_for_hotel(hotel_name, start_date, input_df, output_df, best_model_map, HOTEL_CONFIG):
    """
    Compute recursive SHIFT=0 forecasts for day0..day3 for a single hotel,
    returning [pred0, pred1, pred2, pred3].
    """
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
            pred_val=100
        results_4.append(pred_val)
        predicted_cache[d_] = pred_val

    return results_4

def pishkhor_for_chain(start_date, input_df, output_df, chain_shift_models):
    """
    Recursive SHIFT=0 for the chain, returning [chain_day0, chain_day1, chain_day2, chain_day3].
    """
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

##############################################################################
#                MAIN PAGE: BEST MODELS + AGGREGATION (UI IN FARSI)
##############################################################################
def main_page():
    load_css()
    st.image("tmoble.png", width=180)

    # Refresh button to clear cached data
    if st.button("به روز رسانی"):
        st.cache_data.clear()
        st.success("داده‌ها ریست شدند و مجدداً بارگیری خواهند شد.")

    st.markdown('<div class="header">داشبورد پیش‌بینی</div>', unsafe_allow_html=True)

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "logged_user" not in st.session_state:
        st.session_state.logged_user = None

    system_today = datetime.datetime.now(tehran_tz).date()
    jalali_today = jdatetime.date.fromgregorian(date=system_today)
    greg_str = system_today.strftime("%Y/%m/%d")
    jalali_str = jalali_today.strftime("%Y/%m/%d")
    st.markdown(
        f'<div class="scoreboard">تاریخ میلادی: {greg_str} &nbsp;&nbsp;|&nbsp;&nbsp; تاریخ جلالی: {jalali_str}</div>',
        unsafe_allow_html=True
    )

    # (NEW: the small toggle)
    st.markdown("<div style='text-align: center; margin-bottom: -5px;'><small>سناریو را انتخاب کنید:</small></div>", unsafe_allow_html=True)
    prediction_view_option = st.radio(
        " ",
        ["خوش‌بینانه", "واقع‌بینانه", "بدبینانه"],
        index=1,
        horizontal=True
    )

    # Load main data
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
        blank_val_today = float(input_df.loc[idx_today_input, "Blank"])
    except:
        blank_val_today = 0.0

    match_out = output_df.index[output_df["parsed_output_date"] == system_today].tolist()
    if not match_out:
        st.warning("سطر منطبق در شیت خروجی برای امروز یافت نشد.")
        idx_today_output = None
    else:
        idx_today_output = match_out[0]

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

    # Build holiday flags, WD_, etc. as always. (Unchanged logic)...

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

    holiday_map = {
      "Ramadan_dummy": Ramadan,
      "Moharram_dummy": Moharram,
      "Ashoora_dummy": Ashoora,
      "Arbain_dummy": Arbain,
      "Eid_Fetr_dummy": Fetr,
      "Shabe_Ghadr_dummy": Shabe,
      "Sizdah-be-Dar_dummy": S13,
      "Esfand_dummy": Esfand,
      "Last 5 Days of Esfand_dummy": L5,
      "Norooz_dummy": Nrz,
      "Hol_holiday": HolHol,
      "Hol_none": HolNone,
      "Hol_religious_holiday": HolRel,
      "Yalda_dummy": Yalda
    }

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
        feats.update(holiday_map)
        feats.update(WD_)
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
        feats.update(WD_)
        for c_ in chain_cfg["column_order"]:
            if c_ in holiday_map:
                feats[c_] = holiday_map[c_]
            else:
                feats[c_] = 0

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

    def get_day_label(shift):
        if shift == 0:
            return "امروز"
        elif shift == 1:
            return "فردا"
        elif shift == 2:
            return "پسفردا"
        else:
            return "سه روز بعد"

    pickup_df = get_data_from_pickup_sheet()
    pickup_pivot_df = build_pickup_pivot(pickup_df)

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

        arrival_date_for_shift = datetime.datetime.now(tehran_tz).date() + datetime.timedelta(days=shift)
        pickup_pred = predict_pickup_for_shift(arrival_date_for_shift, pickup_pivot_df, shift)
        if (pickup_pred is not None) and (not np.isnan(pickup_pred)):
            pickup_pred = int(math.ceil(pickup_pred))
        else:
            pickup_pred = 0

        displayed_pred = int(min(int(round(robust)), int(round(future_blank)) - uncertain_val))

        day_results.append({
            "shift": shift,
            "label": get_day_label(shift),
            "روز هفته": week_day,
            "تعداد خالی فعلی": int(round(future_blank)),
            "غیرقطعی": int(uncertain_val),
            "پیش بینی تفکیکی": int(round(sum_houses)),
            "پیش‌بینی کلی": int(round(whole_chain)),
            "پیش بینی نهایی": int(round(robust)),
            "مدل پیکآپ": pickup_pred,
            "پیش بینی نمایشی": displayed_pred,
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

    # ---------------------------------------------------------------------
    # DISPLAY: عدد پیش‌بینی (with toggle)
    # ---------------------------------------------------------------------
    st.subheader("عدد پیش‌بینی")
    cols = st.columns(4)
    pred_gradient = "linear-gradient(135deg, #FFFFFF, #F0F0F0)"
    for idx, (col, row) in enumerate(zip(cols, day_results)):

        # Apply user toggle to select which final prediction to show
        if prediction_view_option == "خوش‌بینانه":
            display_val = row["پیش‌بینی نهایی خوشبینانه"]
        elif prediction_view_option == "بدبینانه":
            display_val = row["پیش‌بینی نهایی بدبینانه"]
        else:
            display_val = row["پیش‌بینی نهایی واقع‌بینانه"]

        extra_content = f"""
        <div id="pred-extra-{idx}" class="extra-text">
        <div>خالی فعلی: {row['تعداد خالی فعلی']}</div>
        <div>غیرقطعی: {row['غیرقطعی']}</div>
        </div>
        """
        
        html_code = f"""
        <html>
        <head>
        <style>
            .score-box {{
            background: {pred_gradient};
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
            <div><b>{row['روز هفته']}</b></div>
            <div><b>پیش‌بینی: {display_val}</b></div>
            {extra_content}
        </div>
        </body>
        </html>
        """
        with col:
            components.html(html_code, height=150, width=200)

    # ---------------------------------------------------------------------
    # DISPLAY: وضعیت بر اساس نرخ اشغال (with toggle)
    # ---------------------------------------------------------------------
    st.subheader("وضعیت بر اساس نرخ اشغال")
    cols = st.columns(4)
    for idx, (col, row) in enumerate(zip(cols, day_results)):
        card_date = datetime.datetime.now(tehran_tz).date() + datetime.timedelta(days=row['shift'])
        target_weekday = card_date.weekday()
        weekday_label = row["روز هفته"] if "روز هفته" in row else ""
        
        avg_month = int(round(compute_avg_for_weekday(input_df, target_weekday, 30)))
        avg_season = int(round(compute_avg_for_weekday(input_df, target_weekday, 90)))
        avg_year = int(round(compute_avg_for_weekday(input_df, target_weekday, 365)))
        
        extra_text = f"""
          {weekday_label}‌های ماه: {avg_month:.0f}<br>
          {weekday_label}‌های فصل: {avg_season:.0f}<br>
          {weekday_label}‌های سال: {avg_year:.0f}
        """

        # same toggle for color
        if prediction_view_option == "خوش‌بینانه":
            pwc = row["پیش‌بینی نهایی خوشبینانه"]
        elif prediction_view_option == "بدبینانه":
            pwc = row["پیش‌بینی نهایی بدبینانه"]
        else:
            pwc = row["پیش‌بینی نهایی واقع‌بینانه"]

        colors_list = [fuzz_color(pwc)]
        final_code = union_fuzzy(colors_list)
        hex_col = color_code_to_hex(final_code)
        gradient = f"linear-gradient(135deg, {hex_col}, {hex_col})"
        text_color = "#333" if hex_col in ["#4A90E2", "#7ED321", "#F5A623"] else "#fff"
        
        html_code = f"""
        <html>
        <head>
        <style>
            .score-box {{
            background: {gradient};
            color: {text_color};
            cursor: pointer;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            width: 100%;
            box-sizing: border-box;
            }}
            .extra-text {{
            display: none;
            margin-top: 10px;
            font-size: 12px;
            font-family: "Tahoma", sans-serif !important;
            }}
        </style>
        <script>
            function toggleExtra_{idx}() {{
                var x = document.getElementById("extra-{idx}");
                if (x.style.display === "none" || x.style.display === "") {{
                    x.style.display = "block";
                }} else {{
                    x.style.display = "none";
                }}
            }}
        </script>
        </head>
        <body>
        <div class="score-box" onclick="toggleExtra_{idx}()">
            <b>{row['label']}</b>
            <div id="extra-{idx}" class="extra-text">
            {extra_text}
            </div>
        </div>
        </body>
        </html>
        """
        with col:
            components.html(html_code, height=150, width=200)

    # (No changes needed for وضعیت بر اساس آمار رزرو)
    st.subheader("وضعیت بر اساس آمار رزرو")
    cols = st.columns(4)
    for idx, (col, row) in enumerate(zip(cols, day_results)):
        arrival_date = datetime.datetime.now(tehran_tz).date() + datetime.timedelta(days=row['shift'])
        count0 = get_pickup_value_for_day(pickup_pivot_df, arrival_date, 0)
        count1 = get_pickup_value_for_day(pickup_pivot_df, arrival_date, 1)
        count2 = get_pickup_value_for_day(pickup_pivot_df, arrival_date, 2)
        count3 = get_pickup_value_for_day(pickup_pivot_df, arrival_date, 3)
        
        if row['shift'] == 0:
            extra_content = f"""
            رزرو همان روز: {count0}<br>
            رزرو از یک روز قبل: {count1}<br>
            رزرو از دو روز قبل: {count2}<br>
            رزرو از سه روز قبل: {count3}
            """
        elif row['shift'] == 1:
            extra_content = f"""
            رزرو از یک روز قبل: {count1}<br>
            رزرو از دو روز قبل: {count2}<br>
            رزرو از سه روز قبل: {count3}
            """
        elif row['shift'] == 2:
            extra_content = f"""
            رزرو از دو روز قبل: {count2}<br>
            رزرو از سه روز قبل: {count3}
            """
        elif row['shift'] == 3:
            extra_content = f"""
            رزرو از سه روز قبل: {count3}
            """
        else:
            extra_content = ""
        
        pickup_val = row['مدل پیکآپ']
        color_val = (pickup_val)
        
        c_code = fuzz_color(color_val)
        final_code = union_fuzzy([c_code])
        hex_col = color_code_to_hex(final_code)
        gradient = f"linear-gradient(135deg, {hex_col}, {hex_col})"
        if hex_col in ["#4A90E2", "#7ED321", "#F5A623"]:
            text_color = "#333"
        else:
            text_color = "#fff"
        
        html_code = f"""
        <html>
        <head>
        <style>
            .score-box {{
            background: {gradient};
            color: {text_color};
            cursor: pointer;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            width: 100%;
            box-sizing: border-box;
            }}
            .extra-text {{
            display: none;
            margin-top: 10px;
            font-size: 12px;
            }}
        </style>
        <script>
            function toggleExtra_{idx}() {{
                var x = document.getElementById("extra-{idx}");
                if (x.style.display === "none" || x.style.display === "") {{
                    x.style.display = "block";
                }} else {{
                    x.style.display = "none";
                }}
            }}
        </script>
        </head>
        <body>
        <div class="score-box" onclick="toggleExtra_{idx}()">
            <b>{row['label']}</b>
            <div id="extra-{idx}" class="extra-text">
            {extra_content}
            </div>
        </div>
        </body>
        </html>
        """
        with col:
            components.html(html_code, height=150, width=200)
    
    st.subheader("مناسبت‌های فصلی ویژه امروز")
    notes = []
    if idx_today_output is not None:
        row_output_today = output_df.loc[idx_today_output]
        def outcol(c):
            return safe_int(row_output_today.get(c, None))
        if (outcol("IsStartOfRamadhan") or outcol("IsMidRamadhan") or outcol("IsEndOfRamadhan")):
            notes.append("ماه رمضان")
        if (outcol("IsStartOfMoharam") or outcol("IsMidMoharam") or outcol("IsEndOfMoharam")):
            notes.append("ماه محرم")
        if outcol("IsHoliday"):
            notes.append("تعطیلات رسمی")
        if outcol("IsTasooaAshoora"):
            notes.append("تاسوعا/عاشورا")
        if outcol("IsArbain"):
            notes.append("اربعین")
        if outcol("IsFetr"):
            notes.append("عید فطر")
        if outcol("IsShabeGhadr"):
            notes.append("شب قدر")
        if outcol("Is13BeDar"):
            notes.append("سیزده به در")
        if (outcol("IsEarlyEsfand") or outcol("IsLateEsfand") or outcol("IsLastDaysOfTheYear")):
            notes.append("اسفند")
        if outcol("IsNorooz"):
            notes.append("نوروز")

    for n in notes:
        st.write(n)
    
    st.write("---")
    st.subheader("مجموعه‌های بحرانی بر اساس پیش‌بینی")
    for day_res in day_results:
        shift = day_res["shift"]
        label = day_res["label"]
        hotel_preds_for_shift = day_res.get("hotel_preds", {})
    
        filtered_hotels = [(h, val) for (h, val) in hotel_preds_for_shift.items()
                           if (not pd.isna(val)) and (val > 3)]
        if not filtered_hotels:
            continue
        
        total_empties = sum(val for _, val in filtered_hotels)
        if total_empties <= 0:
            continue
    
        filtered_hotels.sort(key=lambda x: x[1], reverse=True)
    
        cutoff = 0.8 * total_empties
        cumsum = 0.0
        critical_hotels = []
        for (hname, empties) in filtered_hotels:
            cumsum += empties
            critical_hotels.append((hname, empties))
            if cumsum >= cutoff:
                break
    
        if not critical_hotels:
            continue
    
        lines = [f"**مجموعه‌های بحرانی برای {label}:**\n"]
        
        row_future = idx_today_input + shift
        for (wh, pred_val) in critical_hotels:
            config = HOTEL_CONFIG.get(wh, {})
            cols_for_hotel = config.get("lag_cols", [])
            if (row_future < 0 or row_future >= len(input_df)) or not cols_for_hotel:
                current_empties = 0
            else:
                current_empties = 0
                for c in cols_for_hotel:
                    try:
                        current_empties += float(input_df.loc[row_future, c])
                    except:
                        pass
            
            fa_name = hotel_name_map.get(wh, wh)
            if int(round(current_empties))<=2:
                continue 
            show_val = min(pred_val,current_empties)
            if current_empties - show_val <=2:
                show_val = max(0,show_val - 3)
            if show_val <=2:
                continue
            lines.append(
                f"مجموعه **{fa_name}** با پیش‌بینی **{int(round(show_val))}** خالی برای {label} بحرانی است. "
                f"تعداد خالی فعلی این مجموعه، **{int(round(current_empties))}** است.\n"
            )
    
        final_text = "\n".join(lines)
        st.info(final_text)

    st.write("---")
    st.subheader("ثبت پیش‌بینی خبرگان")

    tmol_pw = st.secrets["tmol_passwords"]
    
    user_passwords = {
        "محمدرضا ایدرم": tmol_pw["idrom"],
        "فرهاد حیدری" : tmol_pw["farhad"],
        "فرشته فرجی":   tmol_pw["fereshte"],
        "آرش پیریایی":  tmol_pw["arash"],
        "فرزین سوری":   tmol_pw["farzin"],
        "احسان همایونی": tmol_pw["ehsan"],
        "امیرحسین محتشم" :tmol_pw["mohtasham"]
    }

    user_column_map = {
        "محمدرضا ایدرم": {
            "columns": [
                "idrom today", "idrom tomorrow", "idrom 2days", "idrom 3days",
                "idrom today reason", "idrom tomorrow reason", "idrom 2days reason", "idrom 3days reason"
            ],
            "time_col": "Idrom Time"
        },
        "فرشته فرجی": {
            "columns": [
                "fereshteh today", "fereshteh tomorrow", "fereshteh 2days", "fereshteh 3days",
                "fereshteh today reason", "fereshteh tomorrow reason", "fereshteh 2days reason", "fereshteh 3days reason"
            ],
            "time_col": "Fereshteh Time"
        },
        "آرش پیریایی": {
            "columns": [
                "Arash today", "Arash tomorrow", "Arash 2days", "Arash 3days",
                "Arash today reason", "Arash tomorrow reason", "Arash 2days reason", "Arash 3days reason"
            ],
            "time_col": "Arash Time"
        },
        "فرزین سوری": {
            "columns": [
                "Farzin today", "Farzin tomorrow", "Farzin 2days", "Farzin 3days",
                "Farzin today reason", "Farzin tomorrow reason", "Farzin 2days reason", "Farzin 3days reason"
            ],
            "time_col": "Farzin Time"
        },
        "احسان همایونی": {
            "columns": [
                "Ehsan today", "Ehsan tomorrow", "Ehsan 2days", "Ehsan 3days",
                "Ehsan today reason", "Ehsan tomorrow reason", "Ehsan 2days reason", "Ehsan 3days reason"
            ],
            "time_col": "Ehsan Time"
        },
         "امیرحسین محتشم": {
            "columns": [
                "Mohtasham today", "Mohtasham tomorrow", "Mohtasham 2days", "Mohtasham 3days",
                "Mohtasham today reason", "Mohtasham tomorrow reason", "Mohtasham 2days reason", "Mohtasham 3days reason"
            ],
            "time_col": "Mohtasham Time"
        },
        "فرهاد حیدری": {
            "columns": [
                "Farhad today", "Farhad tomorrow", "Farhad 2days", "Farhad 3days",
                "Farhad today reason", "Farhad tomorrow reason", "Farhad 2days reason", "Farhad 3days reason"
            ],
            "time_col": "Farhad Time"
        }
    }

    if not st.session_state.logged_in:
        st.markdown("### ورود به سیستم")
        chosen_user = st.selectbox("نام خودتان را از این منو انتخاب کنید", list(user_passwords.keys()))
        typed_pw = st.text_input("رمز عبور:", type="password")
        login_clicked = st.button("تایید هویت")
        if login_clicked:
            if typed_pw == user_passwords.get(chosen_user, ""):
                st.session_state.logged_in = True
                st.session_state.logged_user = chosen_user
                st.success(f"{chosen_user} خوش آمدید! شما می‌توانید پیش‌بینی خود را ثبت کنید.")
                st.button("شروع پیش‌بینی")
            else:
                st.error("رمز عبور اشتباه است!")
    else:
        logout_clicked = st.button("خروج")
        if logout_clicked:
            st.session_state.logged_in = False
            st.session_state.logged_user = None
            st.warning("شما خارج شدید. لطفاً دوباره وارد شوید.")
            st.button("ورود مجدد")
        else:
            st.write(f"شما وارد شده‌اید به عنوان: **{st.session_state.logged_user}**")
            with st.form("prediction_form"):
                day_labels = ["امروز", "فردا", "پسفردا", "سه روز بعد"]
                user_preds = []
                user_reasons = []
                for i, dlabel in enumerate(day_labels):
                    pred_val = st.number_input(f"پیش‌بینی برای {dlabel}", min_value=0, step=1, key=f"user_pred_{i}")
                    reason_val = st.text_input(f"دلیل پیش‌بینی برای {dlabel}", key=f"user_reason_{i}")
                    user_preds.append(pred_val)
                    user_reasons.append(reason_val)
                submit_pred_button = st.form_submit_button("ثبت پیش‌بینی‌ها")
                if submit_pred_button:
                    SCOPES_WRITE = ['https://www.googleapis.com/auth/spreadsheets']
                    service_account_info_write = st.secrets["gcp_service_account"]
                    creds_write = service_account.Credentials.from_service_account_info(
                        service_account_info_write,
                        scopes=SCOPES_WRITE
                    )
                    client_write = gspread.authorize(creds_write)
                    second_spreadsheet_id = "1Pz_zyb7DAz6CnvFrqv77uBP2Z_L7OnjOZkW0dj3m3HY"
                    sheet_write = client_write.open_by_key(second_spreadsheet_id).worksheet("Sheet1")
                    all_records = sheet_write.get_all_records()
                    full_cols = [
                        "Date",
                        "idrom today", "idrom tomorrow", "idrom 2days", "idrom 3days",
                        "idrom today reason", "idrom tomorrow reason", "idrom 2days reason", "idrom 3days reason",
                        "fereshteh today", "fereshteh tomorrow", "fereshteh 2days", "fereshteh 3days",
                        "fereshteh today reason", "fereshteh tomorrow reason", "fereshteh 2days reason", "fereshteh 3days reason",
                        "Arash today", "Arash tomorrow", "Arash 2days", "Arash 3days",
                        "Arash today reason", "Arash tomorrow reason", "Arash 2days reason", "Arash 3days reason",
                        "Farzin today", "Farzin tomorrow", "Farzin 2days", "Farzin 3days",
                        "Farzin today reason", "Farzin tomorrow reason", "Farzin 2days reason", "Farzin 3days reason",
                        "Ehsan today", "Ehsan tomorrow", "Ehsan 2days", "Ehsan 3days",
                        "Ehsan today reason", "Ehsan tomorrow reason", "Ehsan 2days reason", "Ehsan 3days reason",
                        "Mohtasham today", "Mohtasham tomorrow", "Mohtasham 2days", "Mohtasham 3days",
                        "Mohtasham today reason", "Mohtasham tomorrow reason", "Mohtasham 2days reason", "Mohtasham 3days reason",
                        "Farhad today", "Farhad tomorrow", "Farhad 2days", "Farhad 3days",
                        "Farhad today reason", "Farhad tomorrow reason", "Farhad 2days reason", "Farhad 3days reason",
                        "Idrom Time", "Fereshteh Time", "Arash Time", "Farzin Time", "Ehsan Time","Mohtasham Time","Farhad Time"
                    ]
                    df_second = pd.DataFrame(all_records)
                    if df_second.empty:
                        df_second = pd.DataFrame(columns=full_cols)
                    else:
                        for c in full_cols:
                            if c not in df_second.columns:
                                df_second[c] = ""
                    today_str = system_today.strftime("%Y/%m/%d")
                    matching_idx = df_second.index[df_second["Date"] == today_str].tolist()
                    if len(matching_idx) == 0:
                        new_row = {col: "" for col in full_cols}
                        new_row["Date"] = today_str
                        new_row_df = pd.DataFrame([new_row], columns=full_cols)
                        df_second = pd.concat([df_second, new_row_df], ignore_index=True)
                        row_index = df_second.index[-1]
                    else:
                        row_index = matching_idx[0]
                    user_info = user_column_map[st.session_state.logged_user]
                    user_cols = user_info["columns"]
                    for i in range(4):
                        df_second.at[row_index, user_cols[i]] = user_preds[i]
                        df_second.at[row_index, user_cols[i+4]] = user_reasons[i]
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    df_second.at[row_index, user_info["time_col"]] = now_str
                    df_second = df_second[full_cols]
                    data_to_write = df_second.values.tolist()
                    data_to_write.insert(0, df_second.columns.tolist())
                    sheet_write.clear()
                    sheet_write.update("A1", data_to_write)
                    st.success("پیش‌بینی شما با موفقیت ثبت شد.")
                    # ---------------------------------------------------------------------
        # Expert performance table (Sheet2)
        # ---------------------------------------------------------------------
        st.write("---")
        st.subheader("عملکرد پیش‌بینی‌کنندگان")
    
        # 1) Read Sheet2 of the same spreadsheet:
        creds_perf = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        gc_perf = gspread.authorize(creds_perf)
        perf_ws = gc_perf.open_by_key(
            "1Pz_zyb7DAz6CnvFrqv77uBP2Z_L7OnjOZkW0dj3m3HY"
        ).worksheet("Sheet2")
        perf_df = pd.DataFrame(perf_ws.get_all_records())
    
        # 2) Prep
        today_str = system_today.strftime("%Y/%m/%d")
        # find index of today's row (0-based on get_all_records → row2 is index 0)
        idxs = perf_df.index[perf_df["Date"] == today_str].tolist()
        today_idx = idxs[0] if idxs else len(perf_df)-1
        total_days = today_idx + 1  # number of rows from row2 through today
    
        # 3) Define experts and their columns
        experts = {
            "محمدرضا ایدرم":    ("Idrom MSE error",    "Idrom count",    "Idrom timing"),
            "فرشته فرجی":      ("fereshteh MSE error","fereshteh count","fereshteh timing"),
            "آرش پیریایی":     ("Arash MSE error",    "Arash count",    "Arash timing"),
            "فرزین سوری":      ("Farzin MSE error",   "Farzin count",   "Farzin timing"),
            "احسان همایونی":   ("Ehsan MSE error",    "Ehsan count",    "Ehsan timing"),
            "امیرحسین محتشم":  ("Mohtasham MSE error","Mohtasham count","Mohtasham timing")
        }
    
        records = []
        for name, (mse_col, cnt_col, t_col) in experts.items():
            # mean MSE
            mse_avg = perf_df[mse_col].dropna().astype(float).mean()
            # attendance days (from today’s row)
            try:
                attend = int(perf_df.loc[perf_df["Date"] == today_str, cnt_col].iloc[0])
            except:
                attend = 0
            # participation %
            pct = (attend / total_days) if total_days>0 else 0.0
            # mean timing
            timing_avg = perf_df[t_col].dropna().astype(float).mean()
            records.append({
                "نام": name,
                "میانگین خطای پیش‌بینی": mse_avg,
                "تعداد روزهای مشارکت": attend,
                "درصد مشارکت": pct,
                "میانگین سرعت پیش‌بینی": timing_avg
            })
    
        perf = pd.DataFrame(records)
    
        # 4) Derive all metrics
        perf["میانگین دقت پیش‌بینی"] = perf["میانگین خطای پیش‌بینی"] / 340 * 100
        # rank timing (lower is better → rank 1 smallest)
        perf["رتبه سرعت پیش‌بینی"] = perf["میانگین سرعت پیش‌بینی"].rank(method="min")
        # min–max normalize
        eps = 1e-6
        perf["_e"] = (perf["میانگین خطای پیش‌بینی"] - perf["میانگین خطای پیش‌بینی"].min()) \
                     / (perf["میانگین خطای پیش‌بینی"].max() - perf["میانگین خطای پیش‌بینی"].min() + eps)
        perf["_c"] = (perf["تعداد روزهای مشارکت"] - perf["تعداد روزهای مشارکت"].min()) \
                     / (perf["تعداد روزهای مشارکت"].max() - perf["تعداد روزهای مشارکت"].min() + eps)
        perf["_r"] = (perf["رتبه سرعت پیش‌بینی"] - perf["رتبه سرعت پیش‌بینی"].min()) \
                     / (perf["رتبه سرعت پیش‌بینی"].max() - perf["رتبه سرعت پیش‌بینی"].min() + eps)
        perf["نمره نهایی"] = 0.5*perf["_e"] + 0.4*perf["_c"] + 0.1*perf["_r"]
    
        # 5) Final formatting & sort
        perf["درصد مشارکت"] = (perf["درصد مشارکت"]*100).round(1).astype(str) + "%"
        out = perf[[
            "نام",
            "میانگین دقت پیش‌بینی",
            "میانگین خطای پیش‌بینی",
            "تعداد روزهای مشارکت",
            "درصد مشارکت",
            "رتبه سرعت پیش‌بینی",
            "نمره نهایی"
        ]].sort_values("نمره نهایی", ascending=False)
    
        # 6) Render as RTL/Tahoma‐styled HTML table
        st.markdown(
            out.to_html(
                index=False,
                classes="stTable",
                border=0,
                justify="center"
            ),
            unsafe_allow_html=True
        )

                
def main():
    st.set_page_config(page_title="داشبورد پیش‌بینی", page_icon="📈", layout="wide")
    main_page()

if __name__ == "__main__":
    main()
