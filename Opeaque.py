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
    """Create a cached connection to Google Sheets (read-only)."""
    service_account_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build("sheets", "v4", credentials=creds)
    return service

def read_sheet_values(service, spreadsheet_id, sheet_name, cell_range):
    rng = f"{sheet_name}!{cell_range}"
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=rng
    ).execute()
    values = result.get("values", [])
    if not values:
        return pd.DataFrame()
    header = values[0]
    data = values[1:]
    return pd.DataFrame(data, columns=header)

def parse_input_date_str(s):
    try:
        return datetime.datetime.strptime(s.strip(), "%Y/%m/%d").date()
    except:
        return None

def parse_output_date_str(s):
    try:
        return datetime.datetime.strptime(s.strip(), "%A, %B %d, %Y at %I:%M:%S %p").date()
    except:
        return None

def safe_int(val):
    if val is None:
        return 0
    return 1 if str(val).strip() == "1" else 0

##############################################################################
#           FORECAST HELPERS (for SHIFT-based predictions)
##############################################################################

def forecast_univariate_statsmodels(model_fit, shift):
    """One example of SHIFT-based forecast for univariate statsmodels."""
    steps_ahead = shift + 1
    fc = model_fit.forecast(steps=steps_ahead)
    return float(fc.iloc[-1])

def forecast_moving_avg(ma_dict):
    """Another example: returns the average of a stored window."""
    if not isinstance(ma_dict, dict):
        return np.nan
    last_vals = ma_dict.get("last_window", [])
    if len(last_vals) == 0:
        return np.nan
    return float(np.mean(last_vals))

def forecast_ts_decomp_reg(ts_tuple, X_today, shift):
    """For a decomposition + regression approach."""
    decomposition, lr = ts_tuple
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    last_trend = float(trend.dropna().iloc[-1]) if not trend.dropna().empty else 0.0
    seas_vals = seasonal.dropna().values if not seasonal.dropna().empty else [0.0]
    pos = shift % len(seas_vals)
    try:
        resid_pred = float(lr.predict(X_today)[0])
    except:
        resid_pred = 0.0
    return last_trend + seas_vals[pos] + resid_pred

##############################################################################
#                            BASIC CSS
##############################################################################

def load_css():
    st.markdown(
        """
        <style>
        * { font-family: "Tahoma", sans-serif !important; }
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
        }
        .scoreboard {
            background-color: #ecf0f1;
            border: 2px solid #34495e;
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 8px;
            font-size: 16px;
            font-weight: bold;
            color: #34495e;
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
#                SHIFT-BASED: best_model_map + HOTEL_CONFIG
##############################################################################

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

HOTEL_CONFIG = {
   "Ashrafi": {
     "model_prefix": "Ashrafi",
     "lag_cols": ["AshrafiN", "AshrafiS"],
     "column_order": [... ]  # fill in exactly as your original code
   },
   # ... fill out the rest of your configs ...
   "Koroush": {
     "model_prefix": "Koroush",
     "lag_cols": ["Kourosh"],
     "column_order": [ ... ]
   },
   # etc for all hotels
}

##############################################################################
#                SHIFT-BASED PREDICTION UTILS
##############################################################################

def predict_hotel_shift(hotel_name, shift, holiday_map, WD_, idx_today_input, input_df):
    best_model = best_model_map[hotel_name][shift]
    config = HOTEL_CONFIG[hotel_name]
    prefix = config["model_prefix"]
    final_order = config["column_order"]
    lag_cols = config["lag_cols"]

    # Build features
    feats = {}
    feats.update(holiday_map)
    feats.update(WD_)
    # fill lags
    for i in range(1, 16):
        row_i = idx_today_input - i
        feats[f"Lag{i}_EmptyRooms"] = 0.0
        if 0 <= row_i < len(input_df):
            ssum = 0.0
            for c in lag_cols:
                try:
                    ssum += float(input_df.loc[row_i, c])
                except:
                    pass
            feats[f"Lag{i}_EmptyRooms"] = ssum

    row_vals = [feats.get(c, 0.0) for c in final_order]
    X_today = pd.DataFrame([row_vals], columns=final_order)
    model_path = f"results/{prefix}/{best_model}_{prefix}{shift}.pkl"

    # load model
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
    except:
        return np.nan

    # apply model
    if best_model in ["holt_winters","exp_smoothing"]:
        return forecast_univariate_statsmodels(loaded_model, shift)
    elif best_model == "moving_avg":
        return forecast_moving_avg(loaded_model)
    elif best_model == "ts_decomp_reg":
        return forecast_ts_decomp_reg(loaded_model, X_today, shift)
    else:
        try:
            ypred = loaded_model.predict(X_today)
            return float(ypred[0]) if len(ypred) > 0 else np.nan
        except:
            return np.nan


##############################################################################
#                MAIN:  داشبورد فروش اوپک
##############################################################################

def main_page():
    load_css()
    st.markdown('<div class="header">داشبورد فروش اوپک</div>', unsafe_allow_html=True)

    service = create_gsheets_connection()
    SPREADSHEET_ID = "1LI0orqvqci1d75imMfHKxZ512rUUlpA7P1ZYjV-uVO0"

    input_df_raw = read_sheet_values(service, SPREADSHEET_ID, "Input", "A1:ZZ10000")
    output_df_raw = read_sheet_values(service, SPREADSHEET_ID, "Output", "A1:ZZ10000")
    if input_df_raw.empty or output_df_raw.empty:
        st.error("ورودی/خروجی خالی است.")
        return

    input_df_raw["Date"] = input_df_raw.iloc[:,3]
    input_df_raw["Blank"] = input_df_raw.iloc[:,2]
    input_df_raw["parsed_input_date"] = input_df_raw["Date"].apply(parse_input_date_str)

    output_df_raw["parsed_output_date"] = output_df_raw["Date"].apply(parse_output_date_str)

    # SHIFT-based approach
    system_today = datetime.datetime.now(tehran_tz).date()
    idx_matches = input_df_raw.index[input_df_raw["parsed_input_date"] == system_today].tolist()
    if not idx_matches:
        st.warning("سطر امروز یافت نشد.")
        return
    idx_today_input = idx_matches[0]

    # Build a holiday map from the output row, if it exists
    out_idx = output_df_raw.index[output_df_raw["parsed_output_date"]==system_today].tolist()
    if out_idx:
        row_output = output_df_raw.loc[out_idx[0]]
        def outcol(c):
            return safe_int(row_output.get(c, None))
        Ramadan = outcol("IsStartOfRamadhan") or outcol("IsMidRamadhan") or outcol("IsEndOfRamadhan")
        Moharram = outcol("IsStartOfMoharam") or outcol("IsMidMoharam") or outcol("IsEndOfMoharam")
        # etc... fill them out
    else:
        Ramadan=Moharram=0
        # etc

    holiday_map = {
        "Ramadan_dummy": Ramadan,
        "Moharram_dummy": Moharram,
        # fill the rest
    }

    dow = system_today.weekday()
    WD_ = {f"WD_{i}": 1 if i==dow else 0 for i in range(7)}

    # Build SHIFT day_results for days 0..3
    day_results = []
    hotels = list(best_model_map.keys())

    for shift in range(4):
        # sum of SHIFT-based predictions
        hotel_preds = {h: predict_hotel_shift(h, shift, holiday_map, WD_, idx_today_input, input_df_raw)
                       for h in hotels}
        sum_houses = sum(v for v in hotel_preds.values() if not pd.isna(v))

        # get the "future_blank"
        row_fut = idx_today_input + shift
        if row_fut<0 or row_fut>=len(input_df_raw):
            future_blank = 0.0
            uncertain_val = 0.0
        else:
            try:
                future_blank = float(input_df_raw.loc[row_fut,"Blank"])
            except:
                future_blank=0.0
            try:
                uncertain_val = float(input_df_raw.loc[row_fut,"Hold"])
            except:
                uncertain_val=0.0

        # define "بدبینانه" as an example: half the sum or something
        # We'll do a typical approach like robust=0.5*(sum_houses + min(sum_houses,future_blank))
        chain_pred = min(sum_houses, future_blank)
        robust = 0.5*(sum_houses + chain_pred)
        # let's define "بدبینانه" = robust for example
        # in a real scenario, you might define your own logic
        # We'll store it in day_results as "pred_bad"

        # Also gather day label
        label_str = ["امروز","فردا","پسفردا","سه روز بعد"][shift]
        # find "تعداد خالی فعلی" and "غیرقطعی" from the input
        day_results.append({
            "shift": shift,
            "label": label_str,
            "pred_bad": robust,  # "بدبینانه"
            "تعداد خالی فعلی": int(round(future_blank)),
            "غیرقطعی": int(round(uncertain_val)),
            "hotel_preds": hotel_preds
        })

    # Now display the 4 cards for "فروش اوپک پیشنهادی"
    st.subheader("فروش اوپک پیشنهادی")
    cols = st.columns(4)
    for idx, (col, row) in enumerate(zip(cols, day_results)):
        # We do: max(0, row["pred_bad"] - 10)
        raw_val = row["pred_bad"] - 10
        disp_val = max(0, int(round(raw_val)))

        extra_html = f"""
        <div id="extra-{idx}" style="display:none; margin-top:10px; font-size:14px;">
            <div>تعداد خالی فعلی: {row['تعداد خالی فعلی']}</div>
            <div>غیرقطعی: {row['غیرقطعی']}</div>
        </div>
        """
        card_html = f"""
        <html>
        <head>
        <style>
            .card-box {{
                background: linear-gradient(135deg, #FFFFFF, #F0F0F0);
                border-radius: 5px;
                padding: 20px;
                text-align: center;
                cursor: pointer;
            }}
        </style>
        <script>
            function toggleExtra_{idx}() {{
                var x = document.getElementById("extra-{idx}");
                if(!x.style.display || x.style.display==="none") {{
                    x.style.display="block";
                }} else {{
                    x.style.display="none";
                }}
            }}
        </script>
        </head>
        <body>
        <div class="card-box" onclick="toggleExtra_{idx}()">
            <div><b>{row['label']}</b></div>
            <div><b>فروش اوپک پیشنهادی: {disp_val}</b></div>
            {extra_html}
        </div>
        </body>
        </html>
        """
        with col:
            components.html(card_html, height=150)

    st.write("---")
    st.subheader("مجموعه‌های پیشنهادی برای فروش اوپک (80% Coverage)")
    for row in day_results:
        # replicate your Pareto logic
        label = row["label"]
        # Filter hotels with forecast>3
        filtered = [(h, val) for (h,val) in row["hotel_preds"].items() if not pd.isna(val) and val>3]
        if not filtered:
            continue
        total_ = sum(val for (_, val) in filtered)
        if total_<=0:
            continue
        filtered.sort(key=lambda x: x[1], reverse=True)
        cutoff = 0.8*total_
        cumsum=0.0
        chosen=[]
        for (hname, empties) in filtered:
            cumsum += empties
            chosen.append(hname)
            if cumsum>=cutoff:
                break
        if not chosen:
            continue
        # build a dash-str
        dash_str = " - ".join([hotel_name_map.get(x,x) for x in chosen])
        st.info(f"مجموعه‌های پیشنهادی برای فروش اوپک {label}: {dash_str}")


def main():
    st.set_page_config(page_title="داشبورد فروش اوپک", page_icon="📈", layout="wide")
    main_page()

if __name__ == "__main__":
    main()
