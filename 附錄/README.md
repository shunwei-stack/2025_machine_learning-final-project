# Final Project: AI Governance Simulator (Simplified Prototype)

## 簡介
本專案包含一個簡化的城市交通—能源模擬環境，以及兩種決策路徑：
- 若安裝 `stable-baselines3`：訓練並評估 PPO 智能體。
- 若未安裝：使用內建的 FallbackModel（啟發式策略）以保證可執行性。

環境與演算法目標：同時最小化通勤時間與能源排放/消耗。

## 快速開始
1. 建議建立虛擬環境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # 或 venv\\Scripts\\activate (Windows)
   pip install -r requirements.txt
若要使用 PPO，請另外安裝 stable-baselines3。

執行：

bash
複製程式碼
python traffic_energy_rl.py
程式會先執行內建測試，接著訓練或使用 fallback policy，最後進行評估並顯示結果圖。

檔案
traffic_energy_rl.py：主程式（環境、策略、測試、訓練、評估、繪圖）。

requirements.txt：建議安裝套件。

延伸
若要連接更真實的模擬器，可替換環境動態（例如 SUMO、CityLearn）。

可將 fallback policy 換成簡單 Q-learning 或自行實作的離線訓練流程
