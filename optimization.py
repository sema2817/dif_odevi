import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#1.soru
egitim_df = pd.read_csv("egitim.csv")
test_df = pd.read_csv("test.csv")
tahmin_df = pd.read_csv("tahmin_x.csv")

x_train, y_train = egitim_df.iloc[:, 0].values, egitim_df.iloc[:, 1].values
x_test, y_test = test_df.iloc[:, 0].values, test_df.iloc[:, 1].values
x_pred_new = tahmin_df.iloc[:, 1].values

plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color="purple", label="Eğitim Verisi", alpha=0.7)
plt.scatter(x_test, y_test, color="blue", label="Test Verisi", marker="x")
tum_bucuklular = np.arange(-3.0, 3.1, 0.5)
plt.xticks(tum_bucuklular)
plt.xlabel("x")
plt.ylabel("y")
plt.title("1 - Eğitim ve Test Verisinin Dağılımı")
plt.legend()
plt.grid(True)
plt.savefig("soru1_veri.png")
plt.show()

#2.soru
def create_design_matrix(x, degree):
  X = np.ones((len(x), degree + 1))
  for d in range(1, degree + 1):
    X[:, d] = x**d
  return X

def calculate_mse(y_true, y_pred):
  return np.mean((y_true - y_pred) ** 2)

degrees = range(1, 9)
train_mses = []
test_mses = []
deg_to_coef = {}

for deg in degrees:
  X_train = create_design_matrix(x_train, deg)
  X_test = create_design_matrix(x_test, deg)

  beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
  deg_to_coef[deg] = beta

  y_train_pred = X_train @ beta
  y_test_pred = X_test @ beta

  train_mses.append(calculate_mse(y_train, y_train_pred))
  test_mses.append(calculate_mse(y_test, y_test_pred))

results_df = pd.DataFrame(
    {"Derece": list(degrees), "Eğitim MSE": train_mses, "Test MSE": test_mses}
)
print("2 - Derecelere Göre MSE Tablosu ---")
print(results_df.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(degrees, train_mses, marker="o", label="Eğitim MSE", color="purple")
plt.plot(degrees, test_mses, marker="s", label="Test MSE", color="blue")
plt.xlabel("Polinom Derecesi")
plt.ylabel("Ortalama Karesel Hata (MSE)")
plt.title("2 - Derecelere Göre Eğitim ve Test MSE Değişimi")
plt.legend()
plt.grid(True)
plt.savefig("soru2_mse.png")
plt.show()

#3.soru
best_deg = degrees[np.argmin(test_mses)]
print(
    f"\n3 - y için seçilen optimum derece (Test MSE'ye göre): {best_deg}"
)

#4.soru
selected_deg = best_deg
neighbor_degrees = [
    max(1, selected_deg - 1),
    selected_deg,
    min(8, selected_deg + 1),
]

plt.figure(figsize=(10, 6))
for deg in neighbor_degrees:
  X_tr = create_design_matrix(x_train, deg)
  y_tr_pred = X_tr @ deg_to_coef[deg]
  residuals = y_train - y_tr_pred
  plt.scatter(
      x_train, residuals, label=f"Derece {deg}", alpha=0.7, edgecolors="k"
  )

plt.axhline(0, color="red", linestyle="--")
plt.xlabel("x")
plt.ylabel("Kalıntı (Gerçek - Tahmin)")
plt.title("4 - Kalıntı Grafikleri Karşılaştırması")
plt.legend()
plt.grid(True)
plt.savefig("soru4_kalinti.png")
plt.show()

#5.soru
log_y_train = np.log(y_train)
log_train_mse = []
log_test_mse = []
log_coef = {}

for deg in degrees:
  X_train = create_design_matrix(x_train, deg)
  X_test = create_design_matrix(x_test, deg)

  beta_log = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ log_y_train
  log_coef [deg] = beta_log

  y_train_log_pred = np.exp(X_train @ beta_log)
  y_test_log_pred = np.exp(X_test @ beta_log)

  log_train_mse.append(calculate_mse(y_train, y_train_log_pred))
  log_test_mse.append(calculate_mse(y_test, y_test_log_pred))

log_results_df = pd.DataFrame(
    {
        "Derece": list(degrees),
        "Eğitim MSE (Orijinal Ölçek)": log_train_mse,
        "Test MSE (Orijinal Ölçek)": log_test_mse,
    }
)
print("\n 5 - ln(y) Dönüşümlü MSE Tablosu ")
print(log_results_df.to_string(index=False))

best_deg_log = degrees[np.argmin(log_test_mse)]
print(f"5 - ln(y) için seçilen optimum derece: {best_deg_log}")

#6.soru
print("\n 6 - Bölgesel Davranış Analizi ve Görselleştirme")
x_range = np.linspace(-4, 4, 400)

X_range_matrix = create_design_matrix(x_range, best_deg)
y_pred_range_normal = X_range_matrix @ deg_to_coef[best_deg]

X_range_matrix_log = create_design_matrix(x_range, best_deg_log)
y_pred_range_log = np.exp(X_range_matrix_log @ log_coef [best_deg_log])

plt.figure(figsize=(12, 7))
plt.scatter(x_train, y_train, color="purple", label="Eğitim Verisi", alpha=0.6)
plt.plot(
    x_range,
    y_pred_range_normal,
    color="red",
    linewidth=2,
    label=f"Normal Model (Derece {best_deg})",
)
plt.plot(
    x_range,
    y_pred_range_log,
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"ln(y) Dönüşümlü Model (Derece {best_deg_log})",
)

plt.axvspan(
    0.5,
    1.5,
    color="orange",
    alpha=0.2,
    label="İnterpolasyon Bölgesi [0.5, 1.5]",
)
plt.axvspan(3, 4, color="pink", alpha=0.30, label="Ekstrapolasyon [3, 4]")
plt.axvspan(-4, -3, color="pink", alpha=0.30, label="Ekstrapolasyon [-4, -3]")

plt.xlabel("x")
plt.ylabel("y")
plt.title(
    "6 - [-4, 4] Aralığında y ve ln(y) Modellerinin Tahmin Karşılaştırması"
)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig("soru6_karsilastirma.png")
plt.show()

print(
    "İnterpolasyon ve Ekstrapolasyon Yorumları: \n "
    "- İnterpolasyon Bölgesi [0.5, 1.5]: Her iki model de eğitim verisinin"
    " yoğun olduğu bu aralıkta kararlı ve tutarlıdır. \n"
    "- Ekstrapolasyon Bölgeleri [3, 4] ve [-4, -3]: Veri aralığı dışı olduğundan"
    " normal polinom model uçlarda ani salınımlar yapabilir. ln(y) modeli ise"
    " logaritmik dönüşümün baskılayıcı etkisi sayesinde daha stabil davranabilir."
)


#7.soru
def gradient_descent_scaled(X, y, lr=1e-5, epochs=3000):
  X_mean = X[:, 1:].mean(axis=0)
  X_std = X[:, 1:].std(axis=0)
  X_std[X_std == 0] = 1

  X_scaled = X.copy()
  X_scaled[:, 1:] = (X[:, 1:] - X_mean) / X_std

  n_samples, n_features= X_scaled.shape
  beta = np.zeros(n_features)
  cost_history = []

  for _ in range(epochs):
    predictions = X_scaled @ beta
    errors = predictions - y
    cost = (1 / (2 * n_samples)) * np.sum(errors**2)
    cost_history.append(cost)

    grad = (1 / n_samples) * (X_scaled.T @ errors)
    beta -= lr * grad

  return beta, cost_history, X_mean, X_std

X_best_train = create_design_matrix(x_train, best_deg)
beta_gd, cost_hist, x_mean, x_std = gradient_descent_scaled(
    X_best_train, y_train, lr=0.01, epochs=3000
)

plt.figure(figsize=(8, 4))
plt.plot(cost_hist, color="purple")
plt.xlabel("İterasyon")
plt.ylabel("Kayıp (Cost - MSE/2)")
plt.title(
    f"7 - Gradyan Düşümü Kayıp Eğrisi ({best_deg}. Derece) (Ölçeklendirilmiş)"
)
plt.grid(True)
plt.savefig("soru7_kayip.png")
plt.show()

print("\n 7 - Katsayı Karşılaştırması")
print("Analitik Beta:", deg_to_coef[best_deg])
print("Gradyan Düşümü Beta (Ölçeklendirilmiş):", beta_gd)

#8.soru
final_deg = best_deg_log
X_pred_matrix = create_design_matrix(x_pred_new, final_deg)
final_predictions = np.exp(X_pred_matrix @ log_coef [final_deg])

output_df = pd.DataFrame(
    {"id": tahmin_df.iloc[:, 0].values, "y": final_predictions}
)
output_df.to_csv("26011603_tahmin.csv", index=False)
print(
    "\n 8 - Tahminler '26011603_tahmin.csv' dosyasına başarıyla kaydedilmiştir."
)