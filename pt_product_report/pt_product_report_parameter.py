# 精铺推荐ASIN算法参数数据

import numpy as np

# 直发FBM可能性反推
fba_fees_upper = 9  # FBA费用上限
gross_margin_upper = 0.85  # 直发FBM毛利率上限
gross_margin_lower = 0.45  # 直发FBM毛利率下限

# 默认值
fba_fees_rate = 0.15  # 默认FBA费用占比
pre_fees_rate = 0.1  # 默认头程费用占比
product_fees_rate = 0.25  # 默认货值费用占比
referral_fees_rate = 0.15  # 默认佣金占比

gross_rate_upper = 0.5  # 毛利率上限
gross_rate_lower = -0.5  # 毛利率下限

monthly_revenue_C = 600  # C级爬坡基准销额=$600
product_revenue_std = 0.6  # 基准资金利用率为60%
lqs_std = 7.5  # LQS基准=7.5
available_std = 6  # 长期上架基准=6
qa_std = 1  # 基准月均QA数 = 1
revenue_month_C = 4  # C级基准冒出所需月数=4

# --------------------------------蓝海度计算--------------------------------

# 毛估蓝海度计算参数:marting score
MS_a = -1  # 常量a
MS_b = 6  # 常量b
MS_e = 4  # 底数e
MS_c = 0.5  # 中线比c
MS_cs = 1.515  # 中线得分:1-5分，中间分是2.5
# 毛估蓝海度计算公式:    毛估蓝海度 = IFERROR(MROUND(-1+6/(1+POWER(4,-([毛估CPC因子]-1.51*0.5))),0.05),BLANK())

# 转化净值偏差计算参数
CR_p_l = 1  # 价格下限
CR_p_u_uk = 60  # 价格上限_UK
CR_p_u_ot = 150  # 价格上限_其他
CR_c_us = 0.6832  # 系数_US
CR_k_us = 0.3803  # 指数_US
CR_c_uk = 0.3709  # 系数_UK
CR_k_uk = 0.3822  # 指数_UK
CR_c_ot = 0.5127  # 系数_OT
CR_k_ot = 0.2984  # 指数_OT

# 综合蓝海度计算参数
p5 = 1.5  # 广告权重中线:0.75*2
# 转化净值偏差计算公式:    转化净值偏差 = ROUND(DIVIDE([转化净值],0.6832*POWER(MAX(MIN([平均价格],150),1),0.3803)),2)

# 市场蓝海度计算参数
product_acos = 0.6  # 新品ACOS
p_cr_a = 0.5154  # 价格与CR关系系数
p_cr_b = -0.576  # 价格与CR关系指数

# --------------------------------蓝海度计算--------------------------------

regex_pattern_kw = r"pest repellents|gun lubrication|airsoft rifles|airsoft pistols|household batteries|" \
                   r"essential fatty acids|art glues & pastes|mouse pads|vegetable plants & seeds|self defense|" \
                   r"sunglasses|dental universal dental composites|cyanoacrylate adhesives|topical antimicrobials|" \
                   r"flower plants & seeds|sports fan sweatshirts & hoodies|women's fashion hoodies & sweatshirts|" \
                   r"garden fertilizers|christmas trees|bedding duvet cover sets|kids' comforter sets|" \
                   r"electric blankets|mushrooms herbal supplements|blended vitamin & mineral supplements|" \
                   r"aromatherapy diffusers|humidifiers|area rugs|christmas garlands|magnesium mineral supplements|" \
                   r"kitchen rugs|bedding comforter sets|indoor ultrasonic insect & pest repellers|" \
                   r"trace mineral supplements|multivitamins|collagen supplements|christmas tree skirts|tablecloths|" \
                   r"power dental flossers|aromatherapy candles|all-purpose household cleaners|wearable blankets|" \
                   r"herbal & nutritional sleep supplements|end tables|shower curtain sets|" \
                   r"cinnamon herbal supplement|automatic arm blood pressure monitors|handheld vacuums|mattresses|" \
                   r"nicotine patches|sports nutrition testosterone boosters|yo-yos|toy foam blasters|" \
                   r"sports nutrition endurance & energy supplements|bubble blowing solution|squirt guns|" \
                   r"scar reducing treatments|body skin care products|sunglasses|sports nutrition protein bars|" \
                   r"laundry stain removers|liquid laundry detergent|baby care products|ear drops|" \
                   r"detox & cleanse weight loss products|sports nutrition electrolyte replacement drinks|lighters|" \
                   r"detox & cleanse weight loss products|sports nutrition protein bars|medical procedure masks|" \
                   r"sonic bark deterrents|dog repellents|detox & cleanse weight loss products|adhesive heat patches|" \
                   r"sports nutrition testosterone boosters|yeast infection treatments|bedding|kids' bath|" \
                   r"artificial plants & flowers|posters & prints|tea lights|collectible figurines|wall sculptures|" \
                   r"bath rugs|hrow pillow covers|novelty coffee mugs"

# 关联类型清洗
replace_related_type_dict = {'SP': "商品广告", 'FSA': "四星产品", '4+': "四星产品", 'VAV': "看了又看", 'CSI': "相似产品", 'VAP': "看了还看",
                             'AVP': "看了还看", 'MIE': "更多相关", 'COB': "品牌推荐", 'BAV': "看了却买", 'BAB': "买了又买"}

# PMI标签规则

# S-TOP5月均销额
s_sales_bins = [0, 1200, 20000, 60000, 9999999999]
s_sales_labels = [-1, 0, -2, -3]
s_sales_tags = ["天花板不足C+级", "", "天花板A级", "天花板S级"]

# M-价格集中度
m_price_bins = [-9999, -0.5, 0, 9999]
m_price_labels = [11, 10, 0]
m_price_tags = ["定价超分散", "定价分散", ""]

# M-预估平均毛利率
m_gross_bins = [-9999, 0.15, 0.3, 9999]
m_gross_labels = [-0.5, 0, 10]
m_gross_tags = ["预估毛利率低", "", "预估毛利率达标"]

# M-资金利用率
m_revenue_bins = [-9999, 0.6, 1.2, 9999]
m_revenue_labels = [-0.5, 0, 1]
m_revenue_tags = ["资金利用率低", "", "资金利用率高"]

# M-加权FBA运费
m_fba_bins = [16, 30, 9999]
m_fba_labels = [-0.5, -1.5]
m_fba_tags = ["FBA运费高", "FBA运费超高"]

# M-FBM配送占比
m_fbm_bins = [0.2, 0.4, 1]
m_fbm_labels = [0.5, 1]
m_fbm_tags = ["FBM占比略高", "FBM占比高"]

# M-直发FBM月均销额
m_fbm_cal_sales_bins = [600, 1200, 2400, 9999999999]
m_fbm_cal_sales_labels = [0.5, 1, 2]
m_fbm_cal_sales_tags = ["直发FBM月均C级", "直发FBM月均C+级", "直发FBM月均C++级"]

# I-AMZ直营销额占比
i_amz_bins = [0.2, 0.3, 0.5, 1]
i_amz_labels = [-2, -4, -9]
i_amz_tags = ["AMZ直营略多", "AMZ直营多", "AMZ直营过半"]

# I-大牌商标销额占比
i_famous_bins = [0.2, 0.3, 0.5, 1]
i_famous_labels = [-2, -4, -9]
i_famous_tags = ["大牌商标略多", "大牌商标居多", "大牌商标过半"]

# I-中国卖家占比
i_cn_bins = [0, 0.1, 0.25, 0.6, 1]
i_cn_labels = [10, 0.5, 0, -1]
i_cn_tags = ["中国卖家超少", "中国卖家少", "", "中国卖家很多"]

# I-TOP5LQS
i_lqs_top5_bins = [0, 7.5, 8.5, 10]
i_lqs_top5_labels = [1, 0.5, 0]
i_lqs_top5_tags = ["TOP5LQS很低", "TOP5LQS低", ""]

# I-冒出品平均LQS
i_lqs_bins = [0, 8, 9, 10]
i_lqs_labels = [1, 0, -0.5]
i_lqs_tags = ["冒出品LQS低", "", "冒出品LQS高"]

# I-冒出品A+占比
i_ebc_bins = [0, 0.25, 0.4, 0.7, 0.85, 1]
i_ebc_labels = [1, 0.5, 0, -0.5, -1]
i_ebc_tags = ["冒出品A+很少", "冒出品A+少", "", "冒出品A+多", "冒出品A+很多"]

# I-冒出品视频占比
i_video_bins = [0, 0.15, 0.4, 0.6, 1]
i_video_labels = [0.5, 0, -1, -2]
i_video_tags = ["冒出品视频少", "", "冒出品视频多", "冒出品视频超多"]

# I-冒出品QA占比
i_qa_bins = [0, 0.2, 0.7, 1]
i_qa_labels = [0.5, 0, -0.5]
i_qa_tags = ["冒出品少QA", "", "冒出品QA多"]

# I-动销品平均星级
i_rating_bins = [0, 3.9, 4.2, 5]
i_rating_labels = [-0.5, 10, 0]
i_rating_tags = ["极易差评", "易差评", ""]

# I-冒出品低星占比
i_rating_rate_labels = [1, 2]
i_rating_rate_tags = ["差评好卖多", "差评好卖很多"]

# L-TOP5销额占比
l_sales_top5_bins = [0.65, 0.8, 1]
l_sales_top5_labels = [-2.5, -6]
l_sales_top5_tags = ["头部垄断高", "头部垄断超高"]

# L-非TOP5月均销额占比
l_sales_rate_bins = [0.5, 0.7, 1]
l_sales_rate_labels = [1, 2]
l_sales_rate_tags = ["偏长尾", "很长尾"]

# L-非TOP5月均销额
l_sales_bins = [0, 200, 600, 1200, 2400, 4000, 8000, 9999999999]
l_sales_labels = [-2, -1, 0, 1, 1.5, 10, -3]
l_sales_tags = ["长尾月均D级", "长尾月均C-级", "", "长尾月均C+级", "长尾月均B-级", " ", "长尾月均B+级"]

# L-变体中位数
l_variations_bins = [5, 8, 10, 9999]
l_variations_labels = [-1, -3, -9]
l_variations_tags = ["变体挑花眼1", "变体挑花眼2", "变体挑花眼3"]

# E-平均开售月数
e_month_bins = [0, 9, 18, 48, 60, 96, 9999]
e_month_labels = [2, 1, 0, -1, -3, -6]
e_month_tags = ["平均开售月数很少", "平均开售月数少", "", "平均开售4年+", "平均开售5年+", "平均开售8年+"]

# E-冒出品新品占比
e_new_bins = [0, 0.2, 0.4, 0.6, 1]
e_new_labels = [-1, 0, 1, 2]
e_new_tags = ["冒出新品少", "", "冒出新品多", "冒出新品很多"]

# E-新品平均月销额
e_new_sales_bins = [0, 200, 600, 4000, 9999999999]
e_new_sales_labels = [-1, -0.5, 0, 10]
e_new_sales_tags = ["新品销额D/E级", "新品销额C-级", "", "新品销额偏小爆款"]

# E-平均星数
e_ratings_bins = [0, 200, 400, 1000, 2000, 4000, 9999999999]
e_ratings_labels = [1, 0.5, 0, -1, -3, -9]
e_ratings_tags = ["平均星数很少", "平均星数少", "", "平均星数破千", "平均星数破2K", "平均星数破4K"]

# E-少评好卖
e_ratings_low_labels = [0.5, 1]
e_ratings_low_tags = ["少评好卖", "少评好卖2"]

# E-三标新品占比
e_new_lables_bins = [0, 0.2, 1]
e_new_lables_labels = [0, 0.5]
e_new_lables_tags = ["", "三标新品多"]

# E-月销增长率
e_sales_bins = [-9999, -0.2, 0.5, 9999]
e_sales_labels = [-0.5, 0, 10]
e_sales_tags = ["月销负增长", "", "月销增长高"]

# PMI得分
pmi_bins = [-9999, -5, -2, 2.5, 5, 9999]
pmi_labels = [-2, -1, 0, 1, 2]

# PMI_list
pmi_list = ['TOP5月均销额', '价格集中度', '预估平均毛利率', '预估平均资金利用率', '加权FBA运费', 'FBM配送占比', '直发FBM产品占比', '直发FBM月均销额', 'AMZ直营销额占比',
            '大牌商标销额占比', '中国卖家占比', 'TOP5平均LQS', '冒出品平均LQS', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比', '动销品平均星级', '冒出品低星占比',
            'TOP5销额占比', '非TOP5销额占比', '非TOP5月均销额', '动销品变体中位数', '平均开售月数', '冒出品新品占比', '新品平均月销额', '平均星数', '三标新品占比', '月销增长率']
p_list = ['资金利用率高', 'FBM占比略高', 'FBM占比高', '直发FBM多', '直发FBM月均C级', '直发FBM月均C+级', '直发FBM月均C++级', '中国卖家少', 'TOP5LQS低',
          'TOP5LQS很低', '冒出品LQS低', '冒出品A+少', '冒出品A+很少', '冒出品视频少', '冒出品少QA', '差评好卖多', '差评好卖很多', '偏长尾', '很长尾', '长尾月均C+级',
          '长尾月均B-级', '平均开售月数少', '平均开售月数很少', '冒出新品多', '冒出新品很多', '平均星数少', '平均星数很少', '少评好卖', '少评好卖2', '三标新品多']
m_list = ['天花板不足C+级', '天花板A级', '天花板S级', '预估毛利率低', '资金利用率低', 'FBA运费高', 'FBA运费超高', 'AMZ直营过半', 'AMZ直营多', 'AMZ直营略多',
          '大牌商标过半', '大牌商标居多', '大牌商标略多', '中国卖家很多', '冒出品LQS高', '冒出品A+多', '冒出品A+很多', '冒出品视频多', '冒出品视频超多', '冒出品QA多', '极易差评',
          '头部垄断高', '头部垄断超高', '长尾月均C-级', '长尾月均D级', '长尾月均B+级', '变体挑花眼1', '变体挑花眼2', '变体挑花眼3', '平均开售4年+', '平均开售5年+',
          '平均开售8年+', '冒出新品少', '新品销额C-级', '新品销额D/E级', '平均星数破千', '平均星数破2K', '平均星数破4K', '月销负增长']
i_list = ['定价分散', '定价超分散', '预估毛利率达标', '中国卖家超少', '易差评', '新品销额偏小爆款', '月销增长高']

sampling_list = ['价格', '原ASIN推荐度', '相关竞品款数', '有销额竞品款数', '有销额竞品款数占比', '有销额推荐达标款数',
                 '综合竞品推荐度', '达标推荐度占比', 'TOP5月均销额', 'TOP5月销额', '利基月GMV', '价格中位数', '价格集中度',
                 '预估平均毛利率', '预估平均资金利用率', '加权FBA运费', 'FBM配送占比', '直发FBM产品占比', '直发FBM销额占比',
                 '直发FBM月均销额', '广告蓝海度', 'AMZ直营销额占比', '大牌商标销额占比', '中国卖家占比', 'TOP5平均LQS',
                 '冒出品平均LQS', '冒出品A+占比', '冒出品视频占比', '冒出品QA占比', '动销品平均星级', '冒出品低星占比',
                 'TOP5销额占比', '非TOP5销额占比', '非TOP5月均销额', '动销品变体中位数', '平均开售月数', '冒出品新品占比',
                 '新品平均月销额', '平均星数', '销级星数比', '三标新品占比', '月销增长率']

# 综合推荐级别
recommend_bins = [-999, -1, 0, 1, 2, 999]
recommend_labels = ['不推荐', '争议多', '待定', '小推荐', '推荐']

# weight值清洗
replace_weight_error_dict = {'maximum weight: ': '',
                             'minimum weight: ': '',
                             'hundredths ': '',
                             'pounds【460g】': 'pounds',
                             'pounds, 500 pounds': '',
                             'pounds 1.28千克': 'pounds',
                             'manufacturer :': '',
                             'ounces11': '',
                             'ounces9': '',
                             'yes': '',
                             ',': '',
                             ':': '',
                             '@': '',
                             '【': '',
                             '】': '',
                             '?': '',
                             'ounces142': '',
                             '磅': ' pounds',
                             'lb': ' pounds',
                             'lbs': ' pounds',
                             'poundss': ' pounds',
                             '盎司': ' ounces',
                             'oz': ' ounces',
                             '千克': ' kg',
                             '公斤': ' kg',
                             'kilo gramsrams gramsrams': ' kg',
                             'kilo gramsrams': ' kg',
                             'kilograms': ' kg',
                             'gramsrams': ' g',
                             'gramsram': ' g',
                             'kg': ' kg',
                             'kilo': ' kg',
                             '克': ' g',
                             'g$': ' g',
                             '毫克': ' mg',
                             'milligrams': 'mg',
                             'milligram': 'mg',
                             'milli$': ' mg',
                             'grams': ' g',
                             'gram': ' g',
                             'item model number ': '',
                             'department ': '',
                             'unspsc code ': '',
                             'heavy': '',
                             'light': '',
                             'li': '',
                             'no': '',
                             'mo': '',
                             'ne': '',
                             'tons': '',
                             '300ft': '',
                             '0/1wt': '',
                             '\u200e': '',
                             '-': ' ',
                             'available in 3 break strengths': '',
                             'fly fishing line': '',
                             'fly ': '',
                             'fish': '',
                             'gramsht': '',
                             'in': '',
                             'available': '',
                             'wei': '',
                             '  ': ' '}

replace_weight_unit_list = ['pounds', 'pound', 'ounces', 'ounce', 'g', 'grams', 'kg', 'kilograms', 'mg', 'milligrams']

replace_weight_dict = {'pounds': '453.592',
                       'pound': '453.592',
                       'ounces': '28.3495',
                       'ounce': '28.3495',
                       'kg': '1000',
                       'kilograms': '1000',
                       'g': '1',
                       'grams': '1',
                       'mg': '0.0001',
                       'milligrams': '0.0001',
                       '': np.nan}

# dimensions值清洗
replace_dimensions_error_dict = {'inches': '',
                                 'inche': '',
                                 '"d': '',
                                 '"w ': '',
                                 '"l ': '',
                                 '"h ': '',
                                 '"th ': '',
                                 '  ': ''}

# 一级类目清洗
replace_category_dict = {'电器': 'appliances',
                         'appliances 电器': 'appliances',
                         'appliances(家电)': 'appliances',
                         '艺术、手工艺和缝纫': 'arts, crafts & sewing',
                         '艺术、工艺品和缝纫': 'arts, crafts & sewing',
                         'arts, crafts & sewing艺术、手工艺和缝纫': 'arts, crafts & sewing',
                         '艺术、工艺和缝纫': 'arts, crafts & sewing',
                         '汽车': 'automotive',
                         '汽車': 'automotive',
                         '汽车的': 'automotive',
                         'automotive   汽车': 'automotive',
                         'automotive  汽车': 'automotive',
                         '远程和应用程序控制的车辆和零件': 'automotive',
                         'heavy duty & commercial vehicle equipment': 'automotive',
                         'lighting assemblies & accessories': 'automotive',
                         'lights & lighting accessories': 'automotive',
                         'motorcycle & powersports': 'automotive',
                         '婴儿用品': 'baby products',
                         'productos para bebé': 'baby products',
                         'beauty & personal care 重试  错误原因': 'baby products',
                         'baby products  婴儿用品': 'baby products',
                         '美容与个人护理': 'beauty & personal care',
                         '美容和个人护理': 'beauty & personal care',
                         '美容及个人护理': 'beauty & personal care',
                         'beauty & personal care美容与个人护理': 'beauty & personal care',
                         'beauty & personal care美容及个人护理': 'beauty & personal care',
                         '手机及配件': 'cell phones & accessories',
                         '数码产品': 'electronics',
                         '电子产品': 'electronics',
                         'electronics  电子学': 'electronics',
                         'shoe, jewelry & watch accessories': 'clothing, shoes & jewelry',
                         '电子学': 'electronics',
                         'handmade products 手工制品': 'handmade products',
                         '手工制品': 'handmade products',
                         '健康与家居': 'health & household',
                         '健康与家庭': 'health & household',
                         '保健和家庭': 'health & household',
                         'health & household健康与家居': 'health & household',
                         'medical supplies & equipment': 'health & household',
                         'mobility & daily living aids': 'health & household',
                         '家居及厨房': 'home & kitchen',
                         '家居与厨房': 'home & kitchen',
                         '家庭和厨房': 'home & kitchen',
                         '家居&厨房': 'home & kitchen',
                         'home & kitchen家居及厨房': 'home & kitchen',
                         'home & kitchen 家居与厨房': 'home & kitchen',
                         'home & kitchen 家庭和厨房': 'home & kitchen',
                         'home & kitchen家居与厨房': 'home & kitchen',
                         'home & kitchen 重试  错误原因': 'home & kitchen',
                         '家居、厨具、家装': 'home & kitchen',
                         'dining & entertaining': 'home & kitchen',
                         'kitchen & dining': 'home & kitchen',
                         'small appliance parts & accessories': 'home & kitchen',
                         '工业与科学': 'industrial & scientific',
                         '工业和科学': 'industrial & scientific',
                         'industrial & scientific工业与科学': 'industrial & scientific',
                         'food service equipment & supplies': 'industrial & scientific',
                         'janitorial & sanitation supplies': 'industrial & scientific',
                         'lab & scientific products': 'industrial & scientific',
                         'restaurant appliances & equipment': 'industrial & scientific',
                         'professional dental supplies': 'industrial & scientific',
                         'instrument accessories': 'musical instruments',
                         '乐器': 'musical instruments',
                         '楽器': 'musical instruments',
                         '办公用品': 'office products',
                         '办公产品': 'office products',
                         'office products 办公产品': 'office products',
                         'office products  办公用品': 'office products',
                         'office electronics': 'office products',
                         '庭院、草坪和花园': 'patio, lawn & garden',
                         '庭院、草坪和花园花园': 'patio, lawn & garden',
                         'patio, lawn & garden天井、草坪和花园': 'patio, lawn & garden',
                         'patio, lawn & garden庭院、草坪和花园': 'patio, lawn & garden',
                         'grills & outdoor cooking': 'patio, lawn & garden',
                         'outdoor power tools': 'patio, lawn & garden',
                         '宠物用品': 'pet supplies',
                         'pet supplies  宠物用品': 'pet supplies',
                         '服装、鞋履和珠宝': 'clothing, shoes & jewelry',
                         '服装，鞋子和珠宝': 'clothing, shoes & jewelry',
                         '服装、鞋子和珠宝': 'clothing, shoes & jewelry',
                         'clothing, shoes & jewelry 重试  错误原因': 'clothing, shoes & jewelry',
                         'clothing, shoes & jewelry服装、鞋履和珠宝': 'clothing, shoes & jewelry',
                         'clothing, shoes & jewelry 服装、鞋类和珠宝': 'clothing, shoes & jewelry',
                         'clothing, shoes & jewelry服装、鞋子和珠宝': 'clothing, shoes & jewelry',
                         '时尚': 'clothing, shoes & jewelry',
                         '服装、鞋履珠宝': 'clothing, shoes & jewelry',
                         '服装、鞋类和珠宝': 'clothing, shoes & jewelry',
                         '衣類、靴、ジュエリー': 'clothing, shoes & jewelry',
                         'shoe, jewelry & watch accessories 服装、鞋子和珠宝': 'clothing, shoes & jewelry',
                         '运动和户外活动': 'sports & outdoors',
                         '运动与户外': 'sports & outdoors',
                         '运动和户外': 'sports & outdoors',
                         'sports & outdoors 运动与户外': 'sports & outdoors',
                         'sports & outdoors运动与户外': 'sports & outdoors',
                         'sports & outdoors 运动及户外': 'sports & outdoors',
                         'hunting & fishing': 'sports & outdoors',
                         'sports & outdoor recreation accessories': 'sports & outdoors',
                         '工具与家居装修': 'tools & home improvement',
                         '工具和家居装修': 'tools & home improvement',
                         '工具和家居改进': 'tools & home improvement',
                         'tools & home improvement工具和家居装修': 'tools & home improvement',
                         'tools & home improvement工具与家居装修': 'tools & home improvement',
                         'tools & home improvement 工具与家居装修': 'tools & home improvement',
                         'back to results': 'tools & home improvement',
                         'power & hand tools': 'tools & home improvement',
                         'power tool parts & accessories': 'tools & home improvement',
                         'safety & security': 'tools & home improvement',
                         '玩具和游戏': 'toys & games',
                         '玩具与游戏': 'toys & games',
                         'toys & games玩具与游戏': 'toys & games',
                         'toys & games 玩具与游戏': 'toys & games',
                         'toys & games玩具和游戏': 'toys & games',
                         'remote & app controlled vehicle parts': 'toys & games',
                         'remote & app controlled vehicles & parts': 'toys & games'}

# 二级类目清洗
replace_category_dict_2 = {'艺术、手工艺和缝纫:珠饰和珠宝制作': 'arts, crafts & sewing:beading & jewelry making',
                           '艺术、手工艺和缝纫:针线 活': 'arts, crafts & sewing:needlework',
                           'lighting assemblies & accessories:headlight assemblies, parts & accessories': 'automotive:lights & lighting accessories',
                           'lights & lighting accessories:lighting assemblies & accessories': 'automotive:lights & lighting accessories',
                           'motorcycle & powersports:parts': 'automotive:motorcycle & powersports',
                           'heavy duty & commercial vehicle equipment:heavy duty & commercial vehicles parts': 'automotive:heavy duty & commercial vehicle equipment',
                           '汽车:摩托车和动力运动': 'automotive:motorcycle & powersports',
                           '汽车:外饰件': 'automotive:exterior accessories',
                           '汽车:内饰配件': 'automotive:interior accessories',
                           'automotive  汽车:tools & equipment工具和设备': 'automotive:tools & equipment',
                           'automotive  汽车:replacement parts  更换零件': 'automotive:replacement parts',
                           '汽车:高性能零件和配件': 'automotive:replacement parts',
                           '汽车:外部配件': 'automotive:exterior accessories',
                           'beauty & personal care美容与个人护理:skin care  皮肤护理': 'beauty & personal care:skin care',
                           '美容和个人护理:工具和配件': 'beauty & personal care:tools & accessories',
                           '美容及个人护理:化妆品': 'beauty & personal care:skin care',
                           'beauty & personal care美容及个人护理:skin care  皮肤护理': 'beauty & personal care:skin care',
                           '美容及个人护理:工具及配件': 'beauty & personal care:tools & accessories',
                           '美容及个人护理:个人护理': 'beauty & personal care:skin care',
                           '美容及个人护理:皮肤护理': 'beauty & personal care:skin care',
                           '数码产品:计算机和配件': 'electronics:computers & accessories',
                           '电子产品:可穿戴技术': 'electronics:wearable technology',
                           'shoe, jewelry & watch accessories:shoe care & accessories': 'clothing, shoes & jewelry:shoe, jewelry & watch accessories',
                           '服装、鞋履和珠宝:女人': 'clothing, shoes & jewelry:women',
                           '服装，鞋子和珠宝:男人': 'clothing, shoes & jewelry:men',
                           '服装、鞋子和珠宝:女性': 'clothing, shoes & jewelry:women',
                           '服装、鞋子和珠宝:新奇及更多': 'clothing, shoes & jewelry:novelty & more',
                           '服装、鞋履和珠宝:女童服装': 'clothing, shoes & jewelry:girls',
                           '服装、鞋履和珠宝:男人': 'clothing, shoes & jewelry:men',
                           '电器:零件和配件': 'electronics:accessories & supplies',
                           'handmade products 手工制品:home & kitchen 家居与厨房': 'handmade products:home & kitchen',
                           'mobility & daily living aids:mobility aids & equipment': 'health & household:medical supplies & equipment',
                           'medical supplies & equipment:mobility & daily living aids': 'health & household:medical supplies & equipment',
                           'mobility & daily living aids:bathroom safety, aids & accessories': 'health & household:medical supplies & equipment',
                           'health & household健康与家居:health care  保健': 'health & household:health care',
                           '健康与家居:医疗用品和设备': 'health & household:medical supplies & equipment',
                           '健康与家居:口腔护理': 'health & household:oral care',
                           'health & household健康与家居:household supplies  家庭用品': 'health & household:household supplies',
                           '健康与家居:保健': 'health & household:health care',
                           'kitchen & dining:small appliance parts & accessories': 'home & kitchen:kitchen & dining',
                           'small appliance parts & accessories:coffee & espresso machine parts & accessories': 'home & kitchen:kitchen & dining',
                           '家居及厨房:家居装饰产品': 'home & kitchen:home décor products',
                           '家居与厨房:床上用品': 'home & kitchen:bedding',
                           'home & kitchen家居与厨房:bath  浴': 'home & kitchen:bath',
                           '家居及厨房:艺术墙': 'home & kitchen:wall art',
                           '家居及厨房:寝具': 'home & kitchen:bedding',
                           'dining & entertaining:dinnerware & serveware': 'home & kitchen:kitchen & dining',
                           '家居与厨房:厨房和餐饮': 'home & kitchen:kitchen & dining',
                           '家居及厨房:厨房和餐厅': 'home & kitchen:kitchen & dining',
                           'home & kitchen家居与厨房:kitchen & dining厨房和餐饮': 'home & kitchen:kitchen & dining',
                           'kitchen & dining:dining & entertaining': 'home & kitchen:kitchen & dining',
                           'home & kitchen家居与厨房:bedding  床上用品': 'home & kitchen:bedding',
                           'home & kitchen家居与厨房:home décor products家居装饰产品': 'home & kitchen:home décor products',
                           '家居与厨房:家居装饰产品': 'home & kitchen:home décor products',
                           '家庭和厨房:铺垫': 'home & kitchen:bedding',
                           '工业与科学:物料搬运产品': 'industrial & scientific:material handling products',
                           'food service equipment & supplies:concession & vending equipment': 'industrial & scientific:food service equipment & supplies',
                           'lab & scientific products:lab instruments & equipment': 'industrial & scientific:lab & scientific products',
                           'janitorial & sanitation supplies:vacuums & floor cleaning machines': 'industrial & scientific:janitorial & sanitation supplies',
                           'food service equipment & supplies:restaurant appliances & equipment': 'industrial & scientific:food service equipment & supplies',
                           'restaurant appliances & equipment:commercial food preparation equipment': 'industrial & scientific:food service equipment & supplies',
                           '工业与科学:专业牙科用品': 'industrial & scientific:professional dental supplies',
                           '工业与科学:液压、气动和管道': 'industrial & scientific:hydraulics, pneumatics & plumbing',
                           '工业与科学:零售店固定装置和设备': 'industrial & scientific:fasteners',
                           'instrument accessories:drum & percussion accessories': 'musical instruments:instrument accessories',
                           'office electronics:printers & accessories': 'office products:office electronics',
                           '办公用品:办公电子产品': 'office products:office electronics',
                           '办公用品:办公和学校用品': 'office products:office & school supplies',
                           '办公用品:办公及学校用品': 'office products:office & school supplies',
                           'outdoor power tools:replacement parts & accessories': 'patio, lawn & garden:outdoor power tools',
                           'grills & outdoor cooking:outdoor cooking tools & accessories': 'patio, lawn & garden:grills & outdoor cooking',
                           '庭院、草坪和花园:除害虫': 'patio, lawn & garden:pest control',
                           '庭院、草坪和花园:户外电力的工具': 'patio, lawn & garden:outdoor power tools',
                           'patio, lawn & garden庭院、草坪和花园:gardening & lawn care园艺和草坪护理': 'patio, lawn & garden:gardening & lawn care',
                           '宠物用品:小狗': 'pet supplies:dogs',
                           '运动和户外活动:运动的': 'sports & outdoors:sports',
                           '运动与户外:体育': 'sports & outdoors:sports',
                           'hunting & fishing:shooting': 'sports & outdoors:hunting & fishing',
                           'sports & outdoor recreation accessories:field, court & rink equipment': 'sports & outdoors:sports & outdoor recreation accessories',
                           '运动与户外:户外休闲': 'sports & outdoors:outdoor recreation',
                           'power & hand tools:power tool parts & accessories': 'tools & home improvement:power & hand tools',
                           'power tool parts & accessories:power finishing tool parts & accessories': 'tools & home improvement:power & hand tools',
                           '工具与家居装修:照明和吊扇': 'tools & home improvement:lighting & ceiling fans',
                           'tools & home improvement工具与家居装修:lighting & ceiling fans照明和吊扇': 'tools & home improvement:lighting & ceiling fans',
                           'safety & security:personal protective equipment': 'tools & home improvement:safety & security',
                           '工具与家居装修:硬件': 'tools & home improvement:hardware',
                           'back to results': 'tools & home improvement›kitchen & bath fixtures',
                           'remote & app controlled vehicle parts:power plant & driveline systems': 'toys & games:remote & app controlled vehicles & parts',
                           'remote & app controlled vehicles & parts:remote & app controlled vehicle parts': 'toys & games:remote & app controlled vehicles & parts',
                           '玩具和游戏:谜题': 'toys & games:puzzles',
                           '玩具和游戏:儿童电子产品': "toys & games:kids' electronics",
                           'toys & games 玩具与游戏:games & accessories 游戏和配件': 'toys & games:games & accessories',
                           '玩具和游戏:毛绒玩具和毛绒玩具': 'toys & games:stuffed animals & plush toys',
                           '玩具和游戏:拼搭玩具': 'toys & games:puzzles',
                           'toys & games玩具和游戏:featured categories  特色类别': 'toys & games:featured categories',
                           '汽车:更换零件': 'automotive:replacement parts',
                           '汽车:工具和设备': 'automotive:tools & equipment',
                           'beauty & personal care美容与个人护理:hair care  头发护理': 'beauty & personal care:hair care',
                           '美容及个人护理:头发护理': 'beauty & personal care:hair care',
                           '服装、鞋履和珠宝:鞋类、珠宝和手表配饰': 'clothing, shoes & jewelry:shoe, jewelry & watch accessories',
                           '服装、鞋履和珠宝:行李箱和旅行装备': 'clothing, shoes & jewelry:luggage & travel gear',
                           '服装、鞋子和珠宝:行李箱和旅行装备': 'clothing, shoes & jewelry:luggage & travel gear',
                           'health & household健康与家居:household supplies  家居用品': 'health & household:household supplies',
                           'health & household健康与家居:medical supplies & equipment医疗用品和设备': 'health & household:medical supplies & equipment',
                           'health & household健康与家居:stationery & gift wrapping supplies文具和礼品包装用品': 'health & household:stationery & gift wrapping supplies',
                           'health & household健康与家居:wellness & relaxation健康与放松': 'health & household:wellness & relaxation',
                           '健康与家居:家居用品': 'health & household:household supplies',
                           '健康与家居:卫生保健': 'health & household:health care',
                           '家居及厨房:季节性装饰': 'home & kitchen:seasonal décor',
                           '家居及厨房:洗澡': 'home & kitchen:bath',
                           '家居与厨房:供暖、制冷和空气质量': 'home & kitchen:heating, cooling & air quality',
                           '家居与厨房:浴': 'home & kitchen:bath',
                           '家庭和厨房:床上用品': 'home & kitchen:bedding',
                           '工业与科学:包装和运输用品': 'industrial & scientific:packaging & shipping supplies',
                           '工业与科学:测试、测量和检查': 'industrial & scientific:test, measure & inspect',
                           '工业与科学:实验室和科学产品': 'industrial & scientific:lab & scientific products',
                           'patio, lawn & garden庭院、草坪和花园:generators & portable power发电机和便携式电源': 'patio, lawn & garden:generators & portable power',
                           'patio, lawn & garden庭院、草坪和花园:outdoor power tools户外电动工具': 'patio, lawn & garden:outdoor power tools',
                           'patio, lawn & garden露台、草坪和花园': 'patio, lawn & garden:outdoor power tools',
                           '庭院、草坪和花园:户外电动工具': 'patio, lawn & garden:outdoor power tools',
                           '庭院、草坪和花园:烧烤和户外烹饪': 'patio, lawn & garden:grills & outdoor cooking',
                           '庭院、草坪和花园:庭院家具及配件': 'patio, lawn & garden:patio furniture & accessories',
                           '庭院、草坪和花园:园艺和草坪护理': 'patio, lawn & garden:gardening & lawn care',
                           '露台、草坪和花园': 'patio, lawn & garden:gardening & lawn care',
                           '宠物用品:狗': 'pet supplies:dogs',
                           '宠物用品:爬行动物和两栖动物': 'pet supplies:reptiles & amphibians',
                           '运动与户外:狩猎和捕鱼': 'sports & outdoors:hunting & fishing',
                           '运动与户外:运动的': 'sports & outdoors:sports & fitness',
                           'tools & home improvement工具与家居装修:building supplies   建筑用品': 'tools & home improvement:building supplies',
                           'tools & home improvement工具与家居装修:power & hand tools电动和手动工具': 'tools & home improvement:power & hand tools',
                           '工具与家居装修:电动和手动工具': 'tools & home improvement:power & hand tools',
                           '工具与家居装修:建筑用品': 'tools & home improvement:building supplies',
                           'toys & games玩具和游戏:preschool  幼稚园': 'toys & games:preschool',
                           'toys & games玩具和游戏:puzzles  谜题': 'toys & games:puzzles',
                           'toys & games玩具和游戏:stuffed animals & plush toys毛绒玩具和毛绒玩具': 'toys & games:stuffed animals & plush toys',
                           '玩具和游戏:节日聚会用品': 'toys & games:party supplies',
                           '玩具和游戏:娃娃及配件': 'toys & games:dolls & accessories',
                           '玩具和游戏:学习与教育': 'toys & games:learning & education'}

# PMI标签校正规则

# S-TOP5月均销额
s_sales_bins_correction = [0, 1200, 20000, 60000, 9999999999]
s_sales_labels_correction = [-0.5, 0, -1, -3]
s_sales_tags_correction = ["天花板不足C+级", "", "天花板A级", "天花板S级"]

# M-价格集中度
m_price_bins_correction = [-9999, -0.5, 0, 9999]
m_price_labels_correction = [11, 10, 0]
m_price_tags_correction = ["定价超分散", "定价分散", ""]

# M-预估平均毛利率
m_gross_bins_correction = [-9999, 0.15, 0.3, 9999]
m_gross_labels_correction = [-0.3, 0, 10]
m_gross_tags_correction = ["预估毛利率低", "", "预估毛利率达标"]

# M-资金利用率
m_revenue_bins_correction = [-9999, 0.6, 1.2, 9999]
m_revenue_labels_correction = [-0.5, 0, 1]
m_revenue_tags_correction = ["资金利用率低", "", "资金利用率高"]

# M-加权FBA运费
m_fba_bins_correction = [16, 30, 9999]
m_fba_labels_correction = [-0.2, -1.5]
m_fba_tags_correction = ["FBA运费高", "FBA运费超高"]

# M-FBM配送占比
m_fbm_bins_correction = [0.2, 0.4, 1]
m_fbm_labels_correction = [0.8, 1]
m_fbm_tags_correction = ["FBM占比略高", "FBM占比高"]

# M-直发FBM月均销额
m_fbm_cal_sales_bins_correction = [600, 1200, 2400, 9999999999]
m_fbm_cal_sales_labels_correction = [1, 2, 2.5]
m_fbm_cal_sales_tags_correction = ["直发FBM月均C级", "直发FBM月均C+级", "直发FBM月均C++级"]

# I-AMZ直营销额占比
i_amz_bins_correction = [0.2, 0.3, 0.5, 1]
i_amz_labels_correction = [-1.5, -2.5, -4]
i_amz_tags_correction = ["AMZ直营略多", "AMZ直营多", "AMZ直营过半"]

# I-大牌商标销额占比
i_famous_bins_correction = [0.2, 0.3, 0.5, 1]
i_famous_labels_correction = [-2, -3, -6]
i_famous_tags_correction = ["大牌商标略多", "大牌商标居多", "大牌商标过半"]

# I-中国卖家占比
i_cn_bins_correction = [0, 0.1, 0.25, 0.6, 1]
i_cn_labels_correction = [10, 0.8, 0, -1]
i_cn_tags_correction = ["中国卖家超少", "中国卖家少", "", "中国卖家很多"]

# I-TOP5LQS
i_lqs_top5_bins_correction = [0, 7.5, 8.5, 10]
i_lqs_top5_labels_correction = [1.5, 0.8, 0]
i_lqs_top5_tags_correction = ["TOP5LQS很低", "TOP5LQS低", ""]

# I-冒出品平均LQS
i_lqs_bins_correction = [0, 8, 9, 10]
i_lqs_labels_correction = [1.2, 0, -0.5]
i_lqs_tags_correction = ["冒出品LQS低", "", "冒出品LQS高"]

# I-冒出品A+占比
i_ebc_bins_correction = [0, 0.25, 0.4, 0.7, 0.85, 1]
i_ebc_labels_correction = [1, 0.8, 0, -0.5, -1]
i_ebc_tags_correction = ["冒出品A+很少", "冒出品A+少", "", "冒出品A+多", "冒出品A+很多"]

# I-冒出品视频占比
i_video_bins_correction = [0, 0.15, 0.4, 0.6, 1]
i_video_labels_correction = [0.8, 0, -1, -1.5]
i_video_tags_correction = ["冒出品视频少", "", "冒出品视频多", "冒出品视频超多"]

# I-冒出品QA占比
i_qa_bins_correction = [0, 0.2, 0.7, 1]
i_qa_labels_correction = [0.6, 0, -0.5]
i_qa_tags_correction = ["冒出品少QA", "", "冒出品QA多"]

# I-动销品平均星级
i_rating_bins_correction = [0, 3.9, 4.2, 5]
i_rating_labels_correction = [-0.8, 10, 0]
i_rating_tags_correction = ["极易差评", "易差评", ""]

# I-冒出品低星占比
i_rating_rate_labels_correction = [1.5, 2]
i_rating_rate_tags_correction = ["差评好卖多", "差评好卖很多"]

# L-TOP5销额占比
l_sales_top5_bins_correction = [0.65, 0.8, 1]
l_sales_top5_labels_correction = [-2.5, -4]
l_sales_top5_tags_correction = ["头部垄断高", "头部垄断超高"]

# L-非TOP5月均销额占比
l_sales_rate_bins_correction = [0.5, 0.7, 1]
l_sales_rate_labels_correction = [1, 2]
l_sales_rate_tags_correction = ["偏长尾", "很长尾"]

# L-非TOP5月均销额
l_sales_bins_correction = [0, 200, 600, 1200, 2400, 4000, 8000, 9999999999]
l_sales_labels_correction = [-1.5, -1, 0, 1, 1.5, 10, -3]
l_sales_tags_correction = ["长尾月均D级", "长尾月均C-级", "", "长尾月均C+级", "长尾月均B-级", " ", "长尾月均B+级"]

# L-变体中位数
l_variations_bins_correction = [5, 8, 10, 9999]
l_variations_labels_correction = [-1, -3, -9]
l_variations_tags_correction = ["变体挑花眼1", "变体挑花眼2", "变体挑花眼3"]

# E-平均开售月数
e_month_bins_correction = [0, 9, 18, 48, 60, 96, 9999]
e_month_labels_correction = [2, 1.3, 0, -1, -2.5, -5]
e_month_tags_correction = ["平均开售月数很少", "平均开售月数少", "", "平均开售4年+", "平均开售5年+", "平均开售8年+"]

# E-冒出品新品占比
e_new_bins_correction = [0, 0.2, 0.4, 0.6, 1]
e_new_labels_correction = [-1, 0, 1, 2]
e_new_tags_correction = ["冒出新品少", "", "冒出新品多", "冒出新品很多"]

# E-新品平均月销额
e_new_sales_bins_correction = [0, 200, 600, 4000, 9999999999]
e_new_sales_labels_correction = [-1, -0.5, 0, 10]
e_new_sales_tags_correction = ["新品销额D/E级", "新品销额C-级", "", "新品销额偏小爆款"]

# E-平均星数
e_ratings_bins_correction = [0, 200, 400, 1000, 2000, 4000, 9999999999]
e_ratings_labels_correction = [3, 1.5, 0, -1, -3, -9]
e_ratings_tags_correction = ["平均星数很少", "平均星数少", "", "平均星数破千", "平均星数破2K", "平均星数破4K"]

# E-少评好卖
e_ratings_low_labels_correction = [0.5, 1]
e_ratings_low_tags_correction = ["少评好卖", "少评好卖2"]

# E-三标新品占比
e_new_lables_bins_correction = [0, 0.2, 1]
e_new_lables_labels_correction = [0, 0.8]
e_new_lables_tags_correction = ["", "三标新品多"]

# E-月销增长率
e_sales_bins_correction = [-9999, -0.2, 0.5, 9999]
e_sales_labels_correction = [-0.5, 0, 10]
e_sales_tags_correction = ["月销负增长", "", "月销增长高"]

# PMI得分
pmi_bins_correction = [-9999, -5, -2, 2.5, 5, 9999]
pmi_labels_correction = [-2, -1, 0, 1, 2]

# 综合推荐级别
recommend_bins_correction = [-999, -1, 0, 1, 2, 999]
recommend_labels_correction = ['不推荐', '争议多', '待定', '小推荐', '推荐']

# 定位相似竞品毛利测算

return_rate = 0.05  # 退货率
referral_fee_rate = 0.15  # 佣金占比
exchange_loss_rate = 0.03  # 汇损
convert_cm_inch = 0.394  # 单位转换：厘米转英寸
convert_g_kg = 1000  # 单位转换：克转千克
convert_volume_kg_us = 6000  # 单位转换：转体积重千克_US
convert_volume_kg_ukde = 5000  # 单位转换：转体积重千克_UKDE
convert_volume_pound = 139  # 单位转换：转体积重磅
convert_kg_pound = 0.0022  # 单位转换：克转磅
# 汇率
exchange_rate_us = 7.11
exchange_rate_uk = 9.26
exchange_rate_de = 7.72
# VAT
vat_rate_us = 0
vat_rate_uk = 0.16
vat_rate_de = 0.18
# 运费
freight_fee_us = 7
freight_fee_uk = 20
freight_fee_de = 23
freight_air_fee_us = 45
freight_air_fee_uk = 50.5
freight_air_fee_de = 52

price_low = 10  # 低价件单价
price_low_de = 11

# 商品尺寸分段_US
"""
先进行单件重量判断，单件重量满足后，判断剩下的最长边、次长边、最短边、长度+围长。
如任何一个条件不满足，则往下判断，直至找到符合条件的商品尺寸分段
"""

# 重量限制(磅)
limit_weight_small_us = 1
limit_weight_big_cal_us = 3
limit_weight_big_us = 20
limit_weight_large_us = 50
limit_weight_large_50_us = 50
limit_weight_large_70_us = 70
limit_weight_large_150_us = 150
# 最长边限制(英寸)
limit_length_max_small_us = 15
limit_length_max_big_us = 18
limit_length_max_large_us = 59
# 次长边限制(英寸)
limit_length_mid_small_us = 12
limit_length_mid_big_us = 14
limit_length_mid_large_us = 33
# 最短边限制(英寸)
limit_length_min_small_us = 0.75
limit_length_min_big_us = 8
limit_length_min_large_us = 33
# 长度+围长限制(英寸)
limit_perimeter_us = 130

size_tag_list_us = ['小号标准尺寸',
                    '大号标准尺寸',
                    '大号大件',
                    '超大尺寸：0 至 50 磅',
                    '超大尺寸：50 到 70 磅（不含 50 磅）',
                    '超大尺寸：70 至 150 磅（不含 70 磅）',
                    '超大尺寸：150 磅以上（不含 150 磅）']

# 小号标准尺寸费用
weight_small_list_us = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
fba_fee_small_list_us = [3.06, 3.15, 3.24, 3.33, 3.43, 3.53, 3.6, 3.65]  # 标准配送费用
fba_fee_low_small_list_us = [2.29, 2.38, 2.47, 2.56, 2.66, 2.76, 2.83, 2.88]  # 低价配送费用

# 大号标准尺寸费用
weight_big_list_us = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
fba_fee_big_list_us = [3.68, 3.9, 4.15, 4.55, 4.99, 5.37, 5.52, 5.77, 5.87, 6.05, 6.21, 6.62]
fba_fee_low_big_list_us = [2.91, 3.13, 3.38, 3.78, 4.22, 4.6, 4.75, 5, 5.1, 5.28, 5.44, 5.85]
fba_fee_big_us = 6.92
fba_fee_low_big_us = 6.15
fba_fee_big_per_us = 0.08

# 大号大件费用
fba_fee_large_us = 9.61
fba_fee_low_large_us = 8.84
fba_fee_large_per_us = 0.38

# 超大尺寸：0 至 50 磅
fba_fee_large_50_us = 26.33
fba_fee_low_large_50_us = 25.56
fba_fee_large_50_per_us = 0.38

# 超大尺寸：50 到 70 磅（不含 50 磅）
fba_fee_large_70_us = 40.12
fba_fee_low_large_70_us = 39.35
fba_fee_large_70_per_us = 0.75

# 超大尺寸：70 至 150 磅（不含 70 磅）
fba_fee_large_150_us = 54.81
fba_fee_low_large_150_us = 54.04
fba_fee_large_150_per_us = 0.75

# 超大尺寸：150 磅以上（不含 150 磅）
fba_fee_larger_150_us = 194.95
fba_fee_low_larger_150_us = 194.18
fba_fee_larger_150_per_us = 0.19

# 商品尺寸分段_UKDE
"""
判断顺序：商品重量→体积重量→最长边→次长边→最短边→围长
备注：小号信封~标准大件，顺序判断，当不符合标准大件后，先进行特殊大件的判断，在进行大号大件的判断
特殊大件：3个条件只要满足其中1个就属于特殊大件
"""

size_tag_list_ukde = ['小号信封',
                      '标准信封',
                      '大号信封',
                      '超大号信封',
                      '小包裹',
                      '标准包裹',
                      '小号大件',
                      '标准大件',
                      '大号大件',
                      '特殊大件']
# 重量限制(千克)
limit_weight_small_envelope_ukde = 0.08
limit_weight_standard_envelope_ukde = 0.46
limit_weight_big_envelope_ukde = 0.96
limit_weight_large_envelope_ukde = 0.96

limit_weight_small_bag_ukde = 3.9
limit_weight_standard_bag_ukde = 11.9

limit_weight_small_ukde = 1.76
limit_weight_standard_ukde = 29.76
limit_weight_big_ukde = 31.5
limit_weight_large_ukde = 60

limit_weight_v_small_ukde = 25.82
limit_weight_v_standard_ukde = 86.4

# 最长边限制(厘米)
limit_length_max_small_envelope_ukde = 20
limit_length_max_standard_envelope_ukde = 33
limit_length_max_big_envelope_ukde = 33
limit_length_max_large_envelope_ukde = 33

limit_length_max_small_bag_ukde = 35
limit_length_max_standard_bag_ukde = 45

limit_length_max_small_ukde = 61
limit_length_max_standard_ukde = 120
limit_length_max_big_ukde = 175

# 次长边限制(厘米)
limit_length_mid_small_envelope_ukde = 15
limit_length_mid_standard_envelope_ukde = 23
limit_length_mid_big_envelope_ukde = 23
limit_length_mid_large_envelope_ukde = 23

limit_length_mid_small_bag_ukde = 25
limit_length_mid_standard_bag_ukde = 34

limit_length_mid_small_ukde = 46
limit_length_mid_standard_ukde = 60

# 最短边限制(厘米)
limit_length_min_small_envelope_ukde = 1
limit_length_min_standard_envelope_ukde = 2.5
limit_length_min_big_envelope_ukde = 4
limit_length_min_large_envelope_ukde = 6

limit_length_min_small_bag_ukde = 12
limit_length_min_standard_bag_ukde = 26

limit_length_min_small_ukde = 46
limit_length_min_standard_ukde = 60

# 长度+围长限制(厘米)
limit_large_perimeter_ukde = 360

# 小号信封费用
weight_small_envelope_ukde = 0.08
fba_fee_small_envelope_uk = 1.71
fba_fee_low_small_envelope_uk = 1.34
fba_fee_small_envelope_de = 1.9
fba_fee_low_small_envelope_de = 1.45

# 标准信封费用
weight_standard_envelope_list_ukde = [0, 0.06, 0.21, 0.46]
fba_fee_standard_envelope_list_uk = [1.89, 2.07, 2.2]
fba_fee_low_standard_envelope_list_uk = [1.52, 1.69, 1.83]
fba_fee_standard_envelope_list_de = [2.09, 2.23, 2.39]
fba_fee_low_standard_envelope_list_de = [1.64, 1.78, 1.94]

# 大号信封费用
weight_big_envelope_ukde = 0.96
fba_fee_big_envelope_uk = 2.73
fba_fee_low_big_envelope_uk = 2.36
fba_fee_big_envelope_de = 2.74
fba_fee_low_big_envelope_de = 2.29

# 超大号信封费用
weight_large_envelope_ukde = 0.96
fba_fee_large_envelope_uk = 2.95
fba_fee_low_large_envelope_uk = 2.59
fba_fee_large_envelope_de = 3.12
fba_fee_low_large_envelope_de = 2.67

# 小包裹费用
weight_small_bag_list_ukde = [0, 0.15, 0.4, 0.9, 1.4, 1.9, 3.9]
fba_fee_small_bag_list_uk = [2.99, 3.01, 3.05, 3.23, 3.58, 5.62]
fba_fee_small_bag_list_de = [3.12, 3.32, 3.7, 4.37, 4.76, 5.97]

weight_low_small_bag_list_ukde = [0, 0.15, 0.4]
fba_fee_low_small_bag_list_uk = [2.61, 2.64]
fba_fee_low_small_bag_list_de = [2.67, 2.87]
weight_low_small_bag_max_ukde = 0.4

# 标准包裹费用
weight_standard_bag_list_ukde = [0, 0.15, 0.4, 0.9, 1.4, 1.9, 2.9, 3.9, 5.9, 8.9, 11.9]
fba_fee_standard_bag_list_uk = [3, 3.16, 3.37, 3.6, 3.9, 5.65, 5.96, 6.13, 6.99, 7.39]
fba_fee_standard_bag_list_de = [3.22, 3.63, 4.11, 4.84, 5.32, 5.98, 6.55, 6.89, 7.44, 7.73]

# 小号大件费用
weight_small_list_ukde = [0, 0.76, 1.26, 1.76]
fba_fee_small_list_uk = [5.32, 6.17, 6.36]
fba_fee_small_list_de = [6.39, 6.41, 6.43]
fba_fee_small_uk = 6.36
fba_fee_small_de = 6.43

# 标准大件费用
weight_standard_list_ukde = [0, 0.76, 1.76, 2.76, 3.76, 4.76, 9.76, 14.76, 19.76, 24.76, 29.76]
fba_fee_standard_list_uk = [6.32, 6.67, 6.82, 6.86, 6.89, 8.24, 8.82, 9.24, 10.24, 10.25]
fba_fee_standard_list_de = [6.46, 6.77, 7.59, 7.65, 7.68, 8.07, 8.79, 9.34, 10.58, 10.59]
fba_fee_standard_uk = 10.25
fba_fee_standard_de = 10.59

# 大号大件费用
weight_big_list_ukde = [0, 4.76, 9.76, 14.76, 19.76, 24.76, 31.5]
fba_fee_big_list_uk = [11.45, 12.52, 13.22, 13.86, 15.08, 15.12]
fba_fee_big_list_de = [9.26, 10.66, 11, 11.63, 12.86, 12.9]
fba_fee_big_uk = 15.12
fba_fee_big_de = 12.9

# 特殊大件费用
weight_large_list_ukde = [0, 20, 30, 40, 50, 60]
fba_fee_large_list_uk = [15.43, 18.48, 19.16, 42.98, 44.25]
fba_fee_large_list_de = [19.98, 27.16, 28.46, 59.97, 61.17]
fba_fee_large_uk = 44.25
fba_fee_large_de = 61.17

fba_fee_per_ukde = 0.01
fba_fee_large_per_uk = 0.37
fba_fee_large_per_de = 0.38

# 产品等级划分
revenue_list = [1, 400, 600, 4000, 20000, 60000, 9999999]
revenue_tag = ['E', 'D', 'C', 'B', 'A', 'S']
revenue_tag_rank = [1, 2, 3, 4, 5, 6]

# 上架时间区间划分
month_available_list = [1, 2, 3, 6, 12, 24, 36, 60, 9999999]
month_available_tag = ['60天', '90天', '半年', '一年', '两年', '三年', '五年', '五年+']
month_available_tag_rank = [1, 2, 3, 4, 5, 6, 7, 8]

# 价格区间划分
price_list = [1, 10, 20, 30, 50, 70, 100, 200, 9999999]
price_tag = ['1-10', '10-20', '20-30', '30-50', '50-70', '70-100', '100-200', '200+']
price_tag_rank = [1, 2, 3, 4, 5, 6, 7, 8]

# 星级区间划分
rating_list = [1, 3, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 10]
rating_tag = ['1-3', '3-3.7', '3.7-3.9', '3.9-4.1', '4.1-4.3', '4.3-4.5', '4.5-4.7', '4.7-5']
rating_tag_rank = [1, 2, 3, 4, 5, 6, 7, 8]

# 星数区间划分
ratings_list = [1, 10, 20, 30, 50, 100, 500, 1000, 9999999]
ratings_tag = ['1-10', '10-20', '20-30', '30-50', '50-100', '100-500', '500-1000', '1000+']
ratings_tag_rank = [1, 2, 3, 4, 5, 6, 7, 8]

# 店铺类型划分
seller_tag = ['大牌/精品店', '小爆款/旗舰店', '精铺店', '直发杂货店', '泛铺FBA', '跟卖店']

# FBA费用计算单位转换
ounce_pound = 0.0625
gram_pound = 0.00220462
kilogram_pound = 2.20462
pound_pound = 1
inche_cm = 2.54

# 可跟卖链接筛选条件
month_available_limit = 12  # 开售月份>12
price_limit_lower = 9  # 价格>9
price_limit_upper = 199  # 价格<199
weight_limit = 2000  # 重量<2kg
ratings_limit = 5  # 星数>=5
rating_limit = 3.9  # 星级>=3.9
fba_fee_limit = 15  # FBA费用<=15
fba_fee_rate_limit = 0.2  # FBA费用占比<=0.2

# 可跟卖推荐性打分
follow_weight_score = 0.2  # 重量得分权重
follow_weight_list = [1, 100, 300, 500, 1000, 2000]  # 重量分值划分区间
follow_weight_label = [5, 4, 3, 2, 1]  # 重量得分区间

follow_rating_score = 0.2  # 评分得分权重
follow_rating_list = [3.9, 4.2, 4.7, 5.5]  # 评分分值划分区间
follow_rating_label = [1, 2, 3]  # 评分得分区间

follow_ratings_score = 0.2  # 评分数得分权重
follow_ratings_list = [5, 50, 100, 500, 1000, 9999]  # 评分数分值划分区间
follow_ratings_label = [1, 2, 3, 4, 5]  # 评分数得分区间

follow_fba_fee_score = 0.2  # FBA费用得分权重
follow_fba_fee_list = [0, 5, 10, 15]  # FBA费用分值划分区间
follow_fba_fee_label = [3, 2, 1]  # FBA费用得分区间

follow_fba_fee_rate_score = 0.2  # FBA费用占比得分权重
follow_fba_fee_rate_list = [0, 0.05, 0.1, 0.15, 2]  # FBA费用占比分值划分区间
follow_fba_fee_rate_label = [4, 3, 2, 1]  # FBA费用占比得分区间

follow_fbm_score = 0.5  # FBM配送方式加分

# 自动筛词API调用
api_url = 'http://coarse-screen.qingkula.com/full_search/'
headers = {
            'Content-Type': 'application/json',
            # 'Authorization': f'Bearer {API_KEY}'  # 如果需要认证
        }