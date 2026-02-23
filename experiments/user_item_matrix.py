import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo ví dụ nhỏ minh họa
data = {
    "M1": [5, None, 2],
    "M2": [None, 3, 5],
    "M3": [4, 4, None],
    "M4": [None, 2, 4]
}

user_item_example = pd.DataFrame(
    data,
    index=["User1", "User2", "User3"]
)

plt.figure(figsize=(6,4))
sns.heatmap(
    user_item_example,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    cbar=False,
    linewidths=0.5,
    linecolor="gray"
)

plt.title("Example User–Item Matrix")
plt.ylabel("Users")
plt.xlabel("Movies")
plt.tight_layout()
plt.show()