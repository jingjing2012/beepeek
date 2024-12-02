# 利基参数数据


# 加权平均FBA计算参数
n_s_6ounce = 0.37  # 数量_小标不超过6盎司
n_s_12ounce = 3.22  # 数量_小标6至12盎司(不含6盎司)
n_s_16ounce = 3.77  # 数量_小标12至16 盎司(不含12盎司)
n_b_6ounce = 3.72  # 数量_大标不超过6盎司
n_b_12ounce = 3.96  # 数量_大标6至12盎司(不含6盎司)
n_b_16ounce = 4.75  # 数量_大标12至16盎司(不含12盎司)
n_b_2pound = 5.40  # 数量_大标1至2磅(不含1磅)
n_b_3pound = 6.08  # 数量_大标2至3磅(不含2磅)
n_b_20pound = 6.44  # 数量_大标3至20磅(不含3磅)
n_bp_20pound = 0.32  # 数量_大标3至20磅(不含3磅)倍率
n_b_70pound = 9.39  # 数量_小号大件0至70磅
n_bp_70pound = 0.40  # 数量_小号大件0至70磅倍率
n_b_o = 86.71  # 数量_其他大件
n_bp_o = 0.83  # 数量_其他大件倍率

# 毛利计算常量
exchange_loss = 0.03  # 汇损
exchange_rate = 6.9  # 汇率
commission = 0.15  # 佣金
default_ratio = 0.15  # 默认货值占比
default_ratio_h = 0.4  # 默认货值最高占比
default_profit = 0.25  # 默认营销前毛利
search_magnification = 2.5  # 直接搜索倍率
shipping_price_f = 14  # 快船海运价格
shipping_price_s = 8  # 慢船海运价格

# 毛估CPC因子计算参数
# 权重p
p1 = 1  # 转化净值偏差PC
p2 = 1  # 评论星级VR
p3 = 1  # 近半年新品成功率NS & 销量潜力得分SP
p4 = 1  # 点击最多商品数CC
# 常数
a = 1  # 转化净值偏差PC
b = 0.35  # 评论星级VR
c = 0.1  # 近半年新品成功率NS & 销量潜力得分SP
d = 0.08  # 点击最多商品数CC
e = 0.2  # 阈值常量
# 幂K
k2 = -1.5  # 搜索广告占比SR
k3 = -0.6  # 平均缺货率AR
k4 = 0.7  # 评论星级VR
k5 = 0.5  # 点击最多商品数CC
# 基准bl
bl1 = 0  # 转化净值偏差PC:基于价格P进行计算
bl2 = 0.84  # 搜索广告占比SR
bl3 = 0.06  # 平均缺货率AR
bl4 = 4.5  # 评论星级VR
bl5 = 0.5  # 近半年新品成功率NS
bl6 = 6  # 销量潜力得分SP
bl7 = 15  # 点击最多商品数CC

'''
# 毛估CPC因子计算公式
f(SR,AR) = IFERROR(POWER(MAX(IF(ISNUMBER([搜索广告占比档]),[搜索广告占比档],[搜索广告占比SR基准bl]),0.2)/[搜索广告占比SR基准bl],IF([搜索广告占比档]>[搜索广告占比SR基准bl],1/[搜索广告占比SR幂k],[搜索广告占比SR幂k]))*POWER((1-MIN(IF(ISNUMBER([平均缺货率]),[平均缺货率],[平均缺货率AR基准bl]),0.8))/(1-[平均缺货率AR基准bl]),[平均缺货率AR幂k]),1)
f(VR) = IFERROR(IF(ISNUMBER([平均产品星级]),MAX([评论星级VR常数a]*POWER(([评论星级VR基准bl]-[平均产品星级])/0.5,[评论星级VR幂k]),-0.2),0),0)
f(NS,SP) = IFERROR([近半年新品成功率NS常数a]*(IF(ISNUMBER([近半年新品成功率]),[近半年新品成功率]/[近半年新品成功率NS基准bl],1)*([销量潜力得分]/[销量潜力得分SP基准bl])-1),0)
f(CC) = IFERROR(IF(ISNUMBER([商品数量]),POWER(MIN([商品数量],150)/[点击最多商品数CC基准bl],[点击最多商品数CC幂k]),1),1)
bf(CC)和f(SR,AR) = [点击最多商品数CC常数a]*(1-[f(CC)]/[f(SR,AR)])
毛估CPC因子 = MAX(ROUND([转化净值偏差PC权重p]*[转化净值偏差]*[f(SR,AR)]+[评论星级VR权重p]*[f(VR)]+[近半年新品成功率NS权重p]*[f(NS,SP)]+[点击最多商品数CC权重p]*[bf(CC)和f(SR,AR)],2),0)
'''
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
CR_c_uk = 0.3709  # 系数_US
CR_k_uk = 0.3822  # 指数_US
CR_c_ot = 0.5127  # 系数_US
CR_k_ot = 0.2984  # 指数_US

# 综合蓝海度计算参数
p5 = 1.5  # 广告权重中线:0.75*2
# 转化净值偏差计算公式:    转化净值偏差 = ROUND(DIVIDE([转化净值],0.6832*POWER(MAX(MIN([平均价格],150),1),0.3803)),2)

# 市场蓝海度计算参数
product_acos = 0.6  # 新品ACOS
p_cr_a = 0.5154  # 价格与CR关系系数
p_cr_b = -0.576  # 价格与CR关系指数

# 不做的利基剔除
# 按类目剔除
regex_pattern_asin_x = r"shampoos|nahrungsergänzung|vitamine|vitamins|supplements|nutrition|cookies|snacks| \
    ^food cupboard|fresheners|sprays$|freeze-dried food|soils|Digestive|Gourmet|:Food:|Treats|Creams|Body Scrubs| \
    Sunscreens|Vitamins|Baby Foods|:Oils$|:Food$|removers$|^bakery|household cleaners|^fresh & chilled|treatments$| \
    lebensmittel & getränke|e-liquids & nachfüllpacks|vitamine, mineralien & ergänzungsmittel|öle:| \
    massageöle, cremes & lotionen|ungezieferschutz|pflanzenschutz & schädlingsbekämpfung:Insekten|öle$| \
    katzen:gesundheit|futter:|Allzweckreiniger|Küchenreiniger|Metallreiniger|Möbel & Holz-Politur|Polsterreiniger| \
    Spezialputzmittel|Reinigungs- & Lösungsmittel|crem|Düfte|medikamente|gesundheitstests|feuchtigkeitspflege"
# 按利基剔除
regex_pattern_niche_x = r"frozen|drinks|nuts|flour|hyaluronic acid|serum|shampoo|hair dye|hair vitamin|head wax| \
    waschmittel|spray|serum|whitening |ubiquinol|pills|pomade|glutathione|pomade|aerogel|moisturizer|cream$|^spray| \
    ^mousse|^ferulic acid|^wrinkle filler|^hand lotion|sunscreen spf 100|conditioner|coconut hair products|scalp mask| \
    remover|oil$|spray$|drops|treatment|ham$|gin$|cheese$|yoghurt|paint$|powder|oil for| oil$| oils$|soup$| \
    japanische süssigkeiten"
# 白名单
regex_pattern_niche = r"warmer|cool|heated|advent|accessor|card|fold|phone|charger|pen$|stand$|ps.$|plug|tool$| \
    tools|wire|bowl|bottle|brush|coiled|cable|topper|switch|gam|^ps|box|organize|holder|container|cover|pad|stand$| \
    nail|birthday|candle|plea|remover|decoration|hat|clamp|filter|binder|brace|keyboard|paper|paper|light$|mascara| \
    mouse|lipstick|kit$|sock|gift|pink|party|favors|marker|chalk|cage|sex|menstrual|bonsai pots|coiled keyboard cable| \
    aircraft powerglide|brush|blind cleaner|breathing exercise device for lungs|red night light|headset|cup|machine| \
    bottle|box|maker|churner|halskrause katze|grill zubehör|mate becher|gemüsekiste|baumschutz|fliegenwedler| \
    lichtschachtabdeckung|allzwecktücher|bodentücher|magnete für fliegengitter|schneckenzaun|drain free| \
    air conditioner|waschmittel dosierhilfe|spinnenfänger|duschhaube|duschschwamm|luffa schwamm|nagelbürste| \
    peelinghandschuh|rückenbürste|rückenschrubber|schlafmütze|waschmittel aufbewahrung|soil blocker| \
    containers for freezing soup"
