import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_local_data():
    """直接加载本地已下载的数据集"""
    file_path = 'data/online_retail.xlsx'

    if os.path.exists(file_path):
        print("找到本地数据集，直接加载...")
        try:
            df = pd.read_excel(file_path)
            print(f"数据集加载成功，包含 {len(df)} 条记录")

            # 数据清洗和预处理
            df = df.dropna(subset=['CustomerID'])
            df['CustomerID'] = df['CustomerID'].astype(int)
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

            # 过滤掉负数量和负价格（这些通常是退货）
            df = df[df['Quantity'] > 0]
            df = df[df['UnitPrice'] > 0]

            print(f"数据处理完成，最终数据量: {len(df)} 条记录")
            return df

        except Exception as e:
            print(f"加载本地数据集失败: {e}")
            return None
    else:
        print("未找到本地数据集，请先下载数据集")
        return None


def safe_plotly_save(fig, filename):
    """安全保存plotly图片"""
    try:
        fig.write_image(filename, width=1000, height=600, scale=2)
        print(f"图片已保存: {filename}")
        return True
    except Exception as e:
        print(f"保存图片失败 {filename}: {e}")
        # 尝试显示图片
        try:
            fig.show()
            print("已显示图片代替保存")
        except:
            print("无法显示图片")
        return False


# 1. 词云图
def create_wordcloud_plot(df):
    """创建词云图"""
    print("生成词云图...")
    text = ' '.join(df['Description'].dropna().astype(str))

    plt.figure(figsize=(15, 10))
    wordcloud = WordCloud(width=1200, height=600, background_color='white',
                          colormap='viridis', max_words=150,
                          relative_scaling=0.5, random_state=42).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('商品描述词云图 - 高频商品展示', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/wordcloud.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# 2. 改进的漏斗图 - 客户购买转化漏斗
def create_funnel_plot(df):
    """创建改进的漏斗图"""
    print("生成改进的漏斗图...")

    # 计算更详细的转化漏斗
    total_customers = df['CustomerID'].nunique()

    # 计算有购买行为的客户
    purchasing_customers = df.groupby('CustomerID')['TotalPrice'].sum()
    active_customers = purchasing_customers[purchasing_customers > 0].count()

    # 计算复购客户（购买次数>1）
    purchase_counts = df.groupby('CustomerID')['InvoiceNo'].nunique()
    repeat_customers = purchase_counts[purchase_counts > 1].count()

    # 计算高价值客户（消费金额前20%）
    high_value_threshold = purchasing_customers.quantile(0.8)
    high_value_customers = purchasing_customers[purchasing_customers >= high_value_threshold].count()

    funnel_data = pd.DataFrame({
        'Stage': ['总客户数', '活跃客户', '复购客户', '高价值客户'],
        'Count': [total_customers, active_customers, repeat_customers, high_value_customers],
        'Percentage': [100,
                       active_customers / total_customers * 100,
                       repeat_customers / total_customers * 100,
                       high_value_customers / total_customers * 100]
    })

    # 创建更美观的漏斗图
    fig = go.Figure(go.Funnel(
        y=funnel_data['Stage'],
        x=funnel_data['Count'],
        textinfo="value+percent initial",
        opacity=0.8,
        marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]},
        textfont={"size": 14},
        connector={"line": {"color": "royalblue", "width": 3}}
    ))

    fig.update_layout(
        title={
            'text': '客户购买行为转化漏斗',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'darkblue'}
        },
        showlegend=False,
        height=600
    )

    return safe_plotly_save(fig, 'images/funnel_plot.png')


# 3. 气泡图
def create_bubble_plot(df):
    """创建气泡图"""
    print("生成气泡图...")
    country_stats = df.groupby('Country').agg({
        'TotalPrice': 'sum',
        'Quantity': 'sum',
        'CustomerID': 'nunique',
        'UnitPrice': 'mean'
    }).reset_index()

    fig = px.scatter(country_stats, x='CustomerID', y='UnitPrice',
                     size='TotalPrice', color='Quantity',
                     hover_name='Country', size_max=60,
                     title='各国销售情况气泡图',
                     labels={'CustomerID': '客户数量', 'UnitPrice': '平均单价'},
                     color_continuous_scale='viridis')
    return safe_plotly_save(fig, 'images/bubble_plot.png')


# 4. 饼图
def create_pie_plot(df):
    """创建饼图"""
    print("生成饼图...")
    country_sales = df.groupby('Country')['TotalPrice'].sum().nlargest(8)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(country_sales)))
    wedges, texts, autotexts = plt.pie(country_sales.values, labels=country_sales.index,
                                       autopct='%1.1f%%', startangle=90, colors=colors,
                                       textprops={'fontsize': 12})

    # 美化文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.title('各国销售额占比饼图', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/pie_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# 5. 柱形图
def create_bar_plot(df):
    """创建柱形图"""
    print("生成柱形图...")
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()

    fig = px.bar(monthly_sales, x='YearMonth', y='TotalPrice',
                 title='月度销售额趋势',
                 labels={'YearMonth': '月份', 'TotalPrice': '销售额(£)'},
                 color='TotalPrice',
                 color_continuous_scale='blues')

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    return safe_plotly_save(fig, 'images/bar_plot.png')


# 6. 改进的箱型图 - 价格分布分析
def create_box_plot(df):
    """创建改进的箱型图"""
    print("生成改进的箱型图...")

    # 选择主要国家进行分析
    top_countries = df['Country'].value_counts().head(8).index
    filtered_df = df[df['Country'].isin(top_countries)]

    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # 左上图：主要国家价格分布箱型图
    sns.boxplot(data=filtered_df, x='Country', y='UnitPrice', ax=ax1, palette='viridis')
    ax1.set_title('主要国家商品价格分布箱型图', fontsize=14, fontweight='bold')
    ax1.set_xlabel('国家', fontsize=12)
    ax1.set_ylabel('商品价格(£)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # 添加统计标注
    country_stats = filtered_df.groupby('Country')['UnitPrice'].describe()
    for i, country in enumerate(top_countries):
        median_price = country_stats.loc[country, '50%']
        ax1.text(i, median_price, f'£{median_price:.1f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 右上图：价格分布小提琴图（显示完整分布）
    sns.violinplot(data=filtered_df, x='Country', y='UnitPrice', ax=ax2, palette='Set2')
    ax2.set_title('主要国家商品价格分布小提琴图', fontsize=14, fontweight='bold')
    ax2.set_xlabel('国家', fontsize=12)
    ax2.set_ylabel('商品价格(£)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # 左下图：价格分位数分析（去除极端值）
    # 过滤掉极端高价格（前1%）
    price_threshold = filtered_df['UnitPrice'].quantile(0.99)
    filtered_df_clean = filtered_df[filtered_df['UnitPrice'] <= price_threshold]

    sns.boxplot(data=filtered_df_clean, x='Country', y='UnitPrice', ax=ax3, palette='coolwarm')
    ax3.set_title('主要国家商品价格分布（去除前1%极端值）', fontsize=14, fontweight='bold')
    ax3.set_xlabel('国家', fontsize=12)
    ax3.set_ylabel('商品价格(£)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)

    # 右下图：价格对数变换箱型图（更好地显示价格分布）
    filtered_df_log = filtered_df.copy()
    filtered_df_log['Log_UnitPrice'] = np.log1p(filtered_df_log['UnitPrice'])  # log(1+x)避免0值

    sns.boxplot(data=filtered_df_log, x='Country', y='Log_UnitPrice', ax=ax4, palette='plasma')
    ax4.set_title('主要国家商品价格分布（对数变换）', fontsize=14, fontweight='bold')
    ax4.set_xlabel('国家', fontsize=12)
    ax4.set_ylabel('商品价格(对数尺度)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)

    # 设置对数刻度的Y轴标签
    log_ticks = [0, 1, 2, 3, 4, 5]  # 对应 e^0=1, e^1=2.7, e^2=7.4, e^3=20, e^4=54.6, e^5=148.4
    log_labels = ['£1', '£3', '£7', '£20', '£55', '£148']
    ax4.set_yticks(log_ticks)
    ax4.set_yticklabels(log_labels)

    plt.tight_layout(pad=3.0)
    plt.savefig('images/box_plot.png', dpi=300, bbox_inches='tight', facecolor='white')

    # 打印详细的统计分析
    print("\n=== 价格分布统计分析 ===")
    for country in top_countries:
        country_data = filtered_df[filtered_df['Country'] == country]['UnitPrice']
        print(f"\n{country}:")
        print(f"  商品数量: {len(country_data):,}")
        print(f"  平均价格: £{country_data.mean():.2f}")
        print(f"  中位数价格: £{country_data.median():.2f}")
        print(f"  价格标准差: £{country_data.std():.2f}")
        print(f"  价格范围: £{country_data.min():.2f} - £{country_data.max():.2f}")
        print(f"  四分位距: £{country_data.quantile(0.75) - country_data.quantile(0.25):.2f}")

    plt.show()


# 7. 改进的仪表盘 - 多指标仪表盘
def create_gauge_plot(df):
    """创建改进的仪表盘"""
    print("生成改进的仪表盘...")

    # 计算多个关键指标
    total_sales = df['TotalPrice'].sum()
    total_customers = df['CustomerID'].nunique()
    total_products = df['StockCode'].nunique()
    avg_order_value = df.groupby('InvoiceNo')['TotalPrice'].sum().mean()

    # 设置目标值
    sales_target = 10000000
    customer_target = 5000
    product_target = 4000
    aov_target = 500

    # 计算完成率
    sales_achievement = min(total_sales / sales_target * 100, 100)
    customer_achievement = min(total_customers / customer_target * 100, 100)
    product_achievement = min(total_products / product_target * 100, 100)
    aov_achievement = min(avg_order_value / aov_target * 100, 100)

    # 创建多指标仪表盘
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=('销售额完成率', '客户数完成率', '商品数完成率', '客单价完成率')
    )

    # 销售额仪表盘
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=sales_achievement,
        domain={'row': 0, 'column': 0},
        title={'text': f"销售额: £{total_sales:,.0f}"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ), row=1, col=1)

    # 客户数仪表盘
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=customer_achievement,
        domain={'row': 0, 'column': 1},
        title={'text': f"客户数: {total_customers}"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ), row=1, col=2)

    # 商品数仪表盘
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=product_achievement,
        domain={'row': 1, 'column': 0},
        title={'text': f"商品数: {total_products}"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkorange"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ), row=2, col=1)

    # 客单价仪表盘
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=aov_achievement,
        domain={'row': 1, 'column': 1},
        title={'text': f"客单价: £{avg_order_value:.2f}"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ), row=2, col=2)

    fig.update_layout(
        title_text="电商业务核心指标仪表盘",
        height=600,
        font={'size': 12}
    )

    return safe_plotly_save(fig, 'images/gauge_plot.png')


# 8. 折线图
def create_line_plot(df):
    """创建折线图"""
    print("生成折线图...")
    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()

    fig = px.line(daily_sales, x='InvoiceDate', y='TotalPrice',
                  title='日销售额趋势分析',
                  labels={'InvoiceDate': '日期', 'TotalPrice': '销售额(£)'})

    fig.update_traces(line=dict(width=3, color='blue'))
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    return safe_plotly_save(fig, 'images/line_plot.png')


# 9. 散点图
def create_scatter_plot(df):
    """创建散点图"""
    print("生成散点图...")
    customer_stats = df.groupby('CustomerID').agg({
        'TotalPrice': 'sum',
        'InvoiceNo': 'nunique',
        'Quantity': 'sum'
    }).reset_index()

    fig = px.scatter(customer_stats, x='InvoiceNo', y='TotalPrice',
                     size='Quantity', color='TotalPrice',
                     title='客户价值分析散点图',
                     labels={'InvoiceNo': '购买次数', 'TotalPrice': '总消费金额(£)'},
                     color_continuous_scale='viridis')
    return safe_plotly_save(fig, 'images/scatter_plot.png')


# 10. 关系网格图 - 商品类别关联分析
def create_relation_grid_plot(df):
    """创建商品类别关联关系网格图"""
    print("生成商品类别关联关系网格图...")

    try:
        # 从商品描述中提取商品类别
        def extract_category(description):
            if pd.isna(description):
                return 'OTHER'
            desc = str(description).upper()
            if any(word in desc for word in ['BAG', 'BAGS', 'PURSE', 'TOTE']):
                return 'BAGS'
            elif any(word in desc for word in ['JEWEL', 'NECKLACE', 'BRACELET', 'RING']):
                return 'JEWELRY'
            elif any(word in desc for word in ['HOME', 'DECORATION', 'CUSHION', 'CANDLE']):
                return 'HOME_DECOR'
            elif any(word in desc for word in ['TOY', 'DOLL', 'GAME', 'PLUSH']):
                return 'TOYS'
            elif any(word in desc for word in ['KITCHEN', 'UTENSIL', 'PAN', 'POT']):
                return 'KITCHEN'
            elif any(word in desc for word in ['STATIONERY', 'PEN', 'NOTEBOOK', 'PENCIL']):
                return 'STATIONERY'
            elif any(word in desc for word in ['CLOTH', 'DRESS', 'SHIRT', 'TROUSERS']):
                return 'CLOTHING'
            else:
                return 'OTHER'

        print("正在提取商品类别...")
        # 应用类别提取
        df_copy = df.copy()
        df_copy['Category'] = df_copy['Description'].apply(extract_category)

        # 检查类别提取结果
        category_counts = df_copy['Category'].value_counts()
        print(f"商品类别分布: {category_counts.to_dict()}")

        # 构建订单-类别矩阵
        print("正在构建订单-类别矩阵...")
        order_category = df_copy.groupby(['InvoiceNo', 'Category'])['Quantity'].sum().unstack(fill_value=0)

        # 检查矩阵形状
        print(f"订单-类别矩阵形状: {order_category.shape}")

        # 计算类别共现矩阵
        print("正在计算类别共现矩阵...")
        co_occurrence = order_category.T.dot(order_category)

        print("共现矩阵:")
        print(co_occurrence)

        # 使用简单的共现频次作为关联强度
        print("使用共现频次作为关联强度...")
        category_pairs = []
        categories = co_occurrence.index.tolist()

        print(f"发现 {len(categories)} 个商品类别")

        # 计算最大共现次数用于归一化
        max_co_occurrence = co_occurrence.values.max()

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i <= j:  # 避免重复计算
                    # 使用简单的共现频次作为关联强度
                    co_occurrence_count = co_occurrence.loc[cat1, cat2]

                    # 归一化到0-100范围
                    if max_co_occurrence > 0:
                        strength = (co_occurrence_count / max_co_occurrence) * 100
                    else:
                        strength = 0

                    category_pairs.append({
                        'Category1': cat1,
                        'Category2': cat2,
                        'CoOccurrence': co_occurrence_count,
                        'Strength': strength
                    })

        # 检查是否成功生成了关系数据
        if not category_pairs:
            print("错误: 未能生成类别关联数据")
            return

        relation_df = pd.DataFrame(category_pairs)
        print(f"成功生成 {len(relation_df)} 条类别关联记录")

        # 创建关系网格图
        plt.figure(figsize=(14, 12))

        # 创建网格矩阵
        categories_sorted = relation_df.groupby('Category1')['Strength'].mean().sort_values(ascending=False).index
        pivot_table = relation_df.pivot_table(values='Strength', index='Category1', columns='Category2', fill_value=0)
        pivot_table = pivot_table.reindex(index=categories_sorted, columns=categories_sorted)

        # 创建热力图样式的网格图
        mask = np.triu(np.ones_like(pivot_table, dtype=bool))  # 创建上三角掩码
        sns.heatmap(pivot_table,
                    mask=mask,
                    annot=True,
                    fmt='.1f',
                    cmap='YlOrRd',
                    square=True,
                    cbar_kws={'label': '关联强度(%)', 'shrink': 0.8},
                    linewidths=0.5,
                    annot_kws={'size': 10, 'weight': 'bold'})

        plt.title('商品类别关联关系网格图\n(基于订单共现频次)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('商品类别', fontsize=12)
        plt.ylabel('商品类别', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig('images/relation_grid_plot.png', dpi=300, bbox_inches='tight', facecolor='white')

        # 打印关联分析结果
        print("\n=== 商品类别关联分析结果 ===")
        strong_relations = relation_df[relation_df['Strength'] > 10].sort_values('Strength', ascending=False)

        if len(strong_relations) > 0:
            print("强关联类别对 (关联强度 > 10%):")
            for _, row in strong_relations.head(10).iterrows():
                print(
                    f"  {row['Category1']} ↔ {row['Category2']}: {row['Strength']:.1f}% (共现: {row['CoOccurrence']}次)")
        else:
            print("未发现强关联类别对 (关联强度 > 10%)")
            print("所有类别关联关系:")
            for _, row in relation_df.sort_values('Strength', ascending=False).head(10).iterrows():
                print(
                    f"  {row['Category1']} ↔ {row['Category2']}: {row['Strength']:.1f}% (共现: {row['CoOccurrence']}次)")

        plt.show()
        print("关系网格图生成成功!")

    except Exception as e:
        print(f"生成关系网格图时出错: {e}")
        import traceback
        traceback.print_exc()
# 11. 雷达图
def create_radar_plot(df):
    """创建雷达图"""
    print("生成雷达图...")
    top_countries = df['Country'].value_counts().head(4).index
    country_metrics = df[df['Country'].isin(top_countries)].groupby('Country').agg({
        'TotalPrice': 'mean',
        'Quantity': 'mean',
        'UnitPrice': 'mean',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique'
    }).reset_index()

    scaler = StandardScaler()
    metrics_scaled = scaler.fit_transform(country_metrics.select_dtypes(include=[np.number]))
    metrics_scaled = pd.DataFrame(metrics_scaled,
                                  columns=country_metrics.select_dtypes(include=[np.number]).columns)
    metrics_scaled['Country'] = country_metrics['Country']

    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange']

    for i, country in enumerate(top_countries):
        country_data = metrics_scaled[metrics_scaled['Country'] == country].iloc[0]
        fig.add_trace(go.Scatterpolar(
            r=country_data.drop('Country').values,
            theta=metrics_scaled.columns.drop('Country'),
            fill='toself',
            name=country,
            line=dict(color=colors[i], width=2)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2, 2])
        ),
        title="主要国家销售能力雷达图对比",
        showlegend=True,
        height=500
    )
    return safe_plotly_save(fig, 'images/radar_plot.png')


# 12. 热力图
def create_heatmap_plot(df):
    """创建热力图"""
    print("生成热力图...")
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['Weekday'] = df['InvoiceDate'].dt.day_name()

    heatmap_data = df.pivot_table(values='TotalPrice', index='Weekday',
                                  columns='Hour', aggfunc='sum', fill_value=0)

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)

    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False,
                cbar_kws={'label': '销售额(£)'})
    plt.title('销售时间分布热力图 - 按星期和小时', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('星期', fontsize=12)
    plt.tight_layout()
    plt.savefig('images/heatmap_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# 13. 预测图
def create_prediction_plot(df):
    """创建预测图"""
    print("生成预测图...")
    monthly_sales = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum().reset_index()
    monthly_sales['InvoiceDate'] = monthly_sales['InvoiceDate'].astype(str)
    monthly_sales['MonthIndex'] = range(len(monthly_sales))

    X = monthly_sales[['MonthIndex']]
    y = monthly_sales['TotalPrice']

    model = LinearRegression()
    model.fit(X, y)

    future_months = len(monthly_sales) + 3
    X_future = pd.DataFrame({'MonthIndex': range(future_months)})
    y_pred = model.predict(X_future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_sales['InvoiceDate'],
        y=monthly_sales['TotalPrice'],
        mode='lines+markers',
        name='历史数据',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=list(monthly_sales['InvoiceDate']) + [f'预测{i + 1}' for i in range(3)],
        y=y_pred,
        mode='lines',
        name='预测趋势',
        line=dict(color='red', width=3, dash='dash')
    ))

    fig.update_layout(
        title='销售额趋势分析与预测',
        xaxis_title='月份',
        yaxis_title='销售额(£)',
        showlegend=True
    )
    return safe_plotly_save(fig, 'images/prediction_plot.png')


# 14. 面积图
def create_area_plot(df):
    """创建面积图"""
    print("生成面积图...")
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_cumulative = df.groupby('YearMonth')['TotalPrice'].sum().cumsum().reset_index()

    fig = px.area(monthly_cumulative, x='YearMonth', y='TotalPrice',
                  title='累计销售额增长趋势',
                  labels={'YearMonth': '月份', 'TotalPrice': '累计销售额(£)'})

    fig.update_traces(line=dict(width=4), fillcolor='rgba(0,100,80,0.2)')
    return safe_plotly_save(fig, 'images/area_plot.png')


# 15. 复合图
def create_composite_plot(df):
    """创建复合图"""
    print("生成复合图...")
    monthly_stats = df.groupby(df['InvoiceDate'].dt.to_period('M')).agg({
        'TotalPrice': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique'
    }).reset_index()
    monthly_stats['InvoiceDate'] = monthly_stats['InvoiceDate'].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=monthly_stats['InvoiceDate'], y=monthly_stats['TotalPrice'],
               name="销售额", marker_color='royalblue', opacity=0.7),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=monthly_stats['InvoiceDate'], y=monthly_stats['InvoiceNo'],
                   name="订单数量", line=dict(color='firebrick', width=3)),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="销售额与订单数量趋势分析",
        xaxis_title="月份",
        height=500
    )
    fig.update_yaxes(title_text="销售额(£)", secondary_y=False)
    fig.update_yaxes(title_text="订单数量", secondary_y=True)

    return safe_plotly_save(fig, 'images/composite_plot.png')


def main():
    """主函数"""
    print("开始大数据可视化项目...")

    # 创建图片目录
    if not os.path.exists('images'):
        os.makedirs('images')

    # 直接加载本地数据
    df = load_local_data()

    if df is None:
        print("无法加载数据集，程序退出")
        return

    # 显示数据基本信息
    print(f"\n数据集信息:")
    print(f"记录数量: {len(df)}")
    print(f"时间范围: {df['InvoiceDate'].min()} 到 {df['InvoiceDate'].max()}")
    print(f"总销售额: {df['TotalPrice'].sum():,.2f}")
    print(f"客户数量: {df['CustomerID'].nunique()}")
    print(f"商品数量: {df['StockCode'].nunique()}")

    # 生成所有可视化图表
    create_wordcloud_plot(df)
    create_funnel_plot(df)
    create_bubble_plot(df)
    create_pie_plot(df)
    create_bar_plot(df)
    create_box_plot(df)
    create_gauge_plot(df)
    create_line_plot(df)
    create_scatter_plot(df)
    create_relation_grid_plot(df)
    create_radar_plot(df)
    create_heatmap_plot(df)
    create_prediction_plot(df)
    create_area_plot(df)
    create_composite_plot(df)

    print("\n所有可视化图表生成完成！")
    print("图片保存在 'images' 文件夹中")


if __name__ == "__main__":
    main()