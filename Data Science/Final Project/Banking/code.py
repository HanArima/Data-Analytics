import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("banking_data.csv")
print(df.head(10))
print(df.describe())
print(df.columns)

# GLOBAL CHANGES FOR THE PLOT
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

'''
# QUESTION 1
fig1 = sns.displot(df, x="age", bins=20, binwidth=4, hue="y", element="step", height=15, aspect=4/3 )
plt.title("Histogram plot of Range of Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig('Displot of Age.png', dpi=200, bbox_inches="tight")
plt.show()


# QUESTION 2
job_types = df['job'].unique()
job_counts = df['job'].value_counts()
print(job_types)
print(job_counts)

plt.figure(figsize=(8, 8))

plt.title("Distribution of Job Types Among Clients")
palette_color = sns.color_palette('flare')
plt.pie(x=job_counts, labels=job_types, colors=palette_color, autopct='%1.1f%%', textprops={'fontsize':10})
plt.savefig("Pie Chart, Job Type Variation.png", dpi=200, bbox_inches="tight")
plt.show()

sns.displot(data=df, x='job', hue='y', palette='mako')
plt.xticks(rotation=90)
plt.title("Job Variation amongst the clients")
plt.xlabel("Job Types")
plt.ylabel("Count")
plt.savefig("Job type variation.png", dpi=200, bbox_inches="tight")
plt.show()'


# QUESTION 3
marital_status_count = df['marital_status'].value_counts()
palette_color = sns.color_palette('viridis')
plt.pie(x=marital_status_count, labels=['married', 'single', 'divorced'], colors=palette_color, autopct='%1.1f%%', textprops={'fontsize':10})
plt.title("Marital Status Distribution")
plt.savefig("Marital Status Distribution.png", dpi=200, bbox_inches="tight")
plt.show()'


# QUESTION 4
palette_color = sns.color_palette('coolwarm')

edu_count = df['education'].value_counts()
edu_type = df['education'].unique()

plt.pie(x=edu_count , labels=['tertiary' ,'secondary', 'unknown' ,'primary'], colors=palette_color, autopct='%1.2f%%', textprops={'fontsize':10})
plt.title("Education Type Distribution")
plt.savefig("Education Type Distribution.png", dpi=200, bbox_inches="tight")
#plt.show()

sns.displot(data=df, x='job', hue='education', palette='Spectral', multiple='stack', height=15 ,aspect=4/3, binwidth=4)
plt.xticks(rotation=90)
plt.title("Distribution of Education for various Job Types")
plt.xlabel("Job Types")
plt.ylabel("Count")
plt.savefig("Education-Job Distribution.png", dpi=200, bbox_inches="tight")
plt.show()'


# QUESTION 5
credit_count = df['default'].value_counts()
#credit_count_type = df['default'].unique()

palette_color = sns.color_palette("ch:s=-.2,r=.6")
plt.pie(x=credit_count, labels=['Credit Default = No', 'Credit Default = Yes'], autopct='%1.2f%%', colors=palette_color, textprops={'fontsize': 10})
plt.title("Candidate and Default Credit")
plt.savefig("Candidate and Default Credit.png", dpi=200, bbox_inches="tight")
plt.show()


# QUESTION 6

plt.figure(figsize=(12, 8))
sns.violinplot(y='job', x='balance', data=df, palette='viridis')
plt.title("Violin Plot of Client Balances by Job Type")
plt.xlabel("Job Type")
plt.ylabel("Balance")
plt.xticks(rotation=90, ha='right') 
plt.savefig("Violin Plot Balance-Job.png", dpi=200, bbox_inches="tight")
plt.tight_layout()

# QUESTION 7

ax = sns.countplot(data=df, x='housing', palette='mako')
plt.title("Count of Housing Loan")
plt.xlabel("Housing Loan? Yes or No")
plt.ylabel("Count")
    
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.0f}',  
                (p.get_x() + p.get_width() / 2., height),  # X and Y coordinates for the text
                ha='center',  
                va='bottom',  
                xytext=(0, -1),  
                textcoords='offset points')
plt.savefig("Countplot for Housing Loan.png", dpi=200, bbox_inches="tight")
plt.show()plt.show()

# QUESTION 8
ax = sns.countplot(data=df, x='loan', palette='mako')
plt.title("Count of Personal Loan")
plt.xlabel("Personal Loan? Yes or No")
plt.ylabel("Count")
    
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.0f}',  
                (p.get_x() + p.get_width() / 2., height),  # X and Y coordinates for the text
                ha='center',  
                va='bottom',  
                xytext=(0, -1),  
                textcoords='offset points')
plt.show()


# QUESTION 9
contact_count = df['contact'].value_counts()
print(contact_count)

palette_color = sns.cubehelix_palette(start=.5, rot=-.5)
plt.pie(x=contact_count, labels=['cellular', 'unknown', 'telephone'], autopct='%1.2f%%', colors=palette_color, textprops={'fontsize': 10})
plt.title("Communication Contact Types")
plt.savefig("Communication Contact Types.png", dpi=200, bbox_inches="tight")
plt.show()'


# QUESTION 9

sns.countplot(x=df['day'],  edgecolor='black' , palette="flare")
plt.title("Distribution of Last Contact Day of the Month")
plt.xlabel("Day of the Month")
plt.ylabel("Frequency")
plt.show()
'

# QUESTION 10
sns.countplot(x='month', data=df, palette="crest")
plt.title("Distribution of Last Contact Month")
plt.xlabel("Month")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# QUESTION 11
plt.figure(figsize=(10, 6))
plt.hist(df['duration'], fill=True, color="lightseagreen")
plt.title("Density Plot of Last Contact Duration")
plt.xlabel("Duration (Seconds)")
plt.ylabel("Density")
plt.show()



# QUESTION 13
plt.figure(figsize=(10, 6))
plt.hist(df['pdays'], bins=20, edgecolor='black')
plt.title("Distribution of Days Since Previous Contact")
plt.xlabel("Days Since Previous Contact")
plt.ylabel("Frequency")
plt.show()

# QUESTION 14 
plt.figure(figsize=(10, 6))
plt.hist(df['previous'], bins=10, edgecolor='black')
plt.title("Distribution of Previous Contacts")
plt.xlabel("Number of Previous Contacts")
plt.ylabel("Frequency")
plt.show()
'''

# QUESTION 12
plt.figure(figsize=(10, 6))
plt.hist(df['campaign'], bins=10, edgecolor='black')
plt.title("Distribution of Contacts During the Campaign")
plt.xlabel("Number of Contacts")
plt.ylabel("Frequency")
plt.show()

