import numpy as np
import pandas as pd

import pt_product_report_parameter as para


# 指标转换及计算_美国站
def value_convert_us(max_length_cm, mid_length_cm, min_length_cm, weight_g, product_value, exchange_rate):
    # 单位转换
    max_length = max_length_cm * para.convert_cm_inch
    mid_length = mid_length_cm * para.convert_cm_inch
    min_length = min_length_cm * para.convert_cm_inch
    weight_kg = weight_g / para.convert_g_kg
    weight_pound = weight_kg * para.convert_kg_pound  # 商品重量
    product_fee = product_value / exchange_rate
    # 指标计算
    perimeter = max_length + 2 * (mid_length + min_length)  # 长度+围长
    weight_volume_kg = max_length_cm * mid_length_cm * min_length_cm / para.convert_volume_kg_us  # 体积重千克
    weight_volume_pound = max_length * mid_length * min_length / para.convert_volume_pound  # 体积重磅,即发货重量
    weight_max_kg = np.fmax(weight_kg, weight_volume_kg)
    weight_max_pound = np.fmax(weight_pound, weight_volume_pound)
    return max_length, mid_length, min_length, weight_pound, perimeter, weight_max_kg, weight_max_pound, product_fee


# 指标转换及计算_英德站
def value_convert_ukde(max_length_cm, mid_length_cm, min_length_cm, weight_g, product_value, exchange_rate):
    # 单位转换
    weight_kg = weight_g / para.convert_g_kg  # 商品重量
    product_fee = product_value / exchange_rate
    # 指标计算
    perimeter = max_length_cm + 2 * (mid_length_cm + min_length_cm)  # 长度+围长
    weight_volume_kg = max_length_cm * mid_length_cm * min_length_cm / para.convert_volume_kg_ukde  # 体积重千克
    weight_max_kg = np.fmax(weight_kg, weight_volume_kg)
    return perimeter, weight_kg, weight_volume_kg, weight_max_kg, product_fee


# 头程计算
def freight_fee_cal(weight_max_kg, convert, convert_air, exchange_rate):
    freight_fee = weight_max_kg * convert / exchange_rate
    freight_fee_air = weight_max_kg * convert_air / exchange_rate
    return freight_fee, freight_fee_air


# 规格打标_美国站
def size_tag_us_cal(max_length, mid_length, min_length, perimeter, weight_max_pound):
    conditions_size_tag_us = [
        (weight_max_pound <= para.limit_weight_small_us) & (max_length <= para.limit_length_max_small_us) & (
                mid_length <= para.limit_length_mid_small_us) & (min_length <= para.limit_length_min_small_us),
        (weight_max_pound <= para.limit_weight_big_us) & (max_length <= para.limit_length_max_big_us) & (
                mid_length <= para.limit_length_mid_big_us) & (min_length <= para.limit_length_min_big_us),
        (weight_max_pound <= para.limit_weight_large_us) & (max_length <= para.limit_length_max_large_us) & (
                mid_length <= para.limit_length_mid_large_us) & (min_length <= para.limit_length_min_large_us) & (
                perimeter <= para.limit_perimeter_us),
        (weight_max_pound <= para.limit_weight_large_50_us),
        (weight_max_pound > para.limit_weight_large_50_us) & (weight_max_pound <= para.limit_weight_large_70_us),
        (weight_max_pound > para.limit_weight_large_70_us) & (weight_max_pound <= para.limit_weight_large_150_us),
        (weight_max_pound > para.limit_weight_large_150_us)
    ]
    size_tag_us = np.select(conditions_size_tag_us, para.size_tag_list_us, default='-')
    return size_tag_us


# 规格打标_英德站
def size_tag_ukde_cal(max_length, mid_length, min_length, perimeter, weight_kg, weight_volume_kg):
    conditions_size_tag_ukde = [
        (weight_kg <= para.limit_weight_small_envelope_ukde) & (
                max_length <= para.limit_length_max_small_envelope_ukde) & (
                mid_length <= para.limit_length_mid_small_envelope_ukde) & (
                min_length <= para.limit_length_min_small_envelope_ukde),
        (weight_kg <= para.limit_weight_standard_envelope_ukde) & (
                max_length <= para.limit_length_max_standard_envelope_ukde) & (
                mid_length <= para.limit_length_mid_standard_envelope_ukde) & (
                min_length <= para.limit_length_min_standard_envelope_ukde),
        (weight_kg <= para.limit_weight_big_envelope_ukde) & (max_length <= para.limit_length_max_big_envelope_ukde) & (
                mid_length <= para.limit_length_mid_big_envelope_ukde) & (
                min_length <= para.limit_length_min_big_envelope_ukde),
        (weight_kg <= para.limit_weight_large_envelope_ukde) & (
                max_length <= para.limit_length_max_large_envelope_ukde) & (
                mid_length <= para.limit_length_mid_large_envelope_ukde) & (
                min_length <= para.limit_length_min_large_envelope_ukde),
        (weight_kg <= para.limit_weight_small_bag_ukde) & (max_length <= para.limit_length_max_small_bag_ukde) & (
                mid_length <= para.limit_length_mid_small_bag_ukde) & (
                min_length <= para.limit_length_min_small_bag_ukde),
        (weight_kg <= para.limit_weight_standard_bag_ukde) & (max_length <= para.limit_length_max_standard_bag_ukde) & (
                mid_length <= para.limit_length_mid_standard_bag_ukde) & (
                min_length <= para.limit_length_min_standard_bag_ukde),
        (weight_kg > para.limit_weight_small_ukde) & (weight_volume_kg > para.limit_weight_v_small_ukde) & (
                max_length <= para.limit_length_max_small_ukde) & (
                mid_length <= para.limit_length_mid_small_ukde) & (min_length <= para.limit_length_min_small_ukde),
        (weight_kg > para.limit_weight_standard_ukde) & (weight_volume_kg > para.limit_weight_v_standard_ukde) & (
                max_length <= para.limit_length_max_standard_ukde) & (
                mid_length <= para.limit_length_mid_standard_ukde) & (
                min_length <= para.limit_length_min_standard_ukde),
        (weight_kg <= para.limit_weight_big_ukde) & (max_length <= para.limit_length_max_big_ukde) & (
                perimeter <= para.limit_large_perimeter_ukde),
        (weight_kg > para.limit_weight_big_ukde) | (max_length > para.limit_length_max_big_ukde) | (
                perimeter > para.limit_large_perimeter_ukde)
    ]
    size_tag_ukde = np.select(conditions_size_tag_ukde, para.size_tag_list_ukde, default='-')
    return size_tag_ukde


# FBA费用计算_US
def fba_fee_us_cal(price, size_tag, weight_pound, weight_max_pound):
    fba_fee_us = 0
    if price >= para.price_low:
        if size_tag == para.size_tag_list_us[0]:
            index = np.digitize([weight_pound], para.weight_small_list_us, right=True)[0] - 1
            fba_fee_us = para.fba_fee_small_list_us[index]
        elif size_tag == para.size_tag_list_us[1]:
            if weight_max_pound <= para.limit_weight_big_cal_us:
                index = np.digitize([weight_max_pound], para.weight_big_list_us, right=True)[0] - 1
                fba_fee_us = para.fba_fee_big_list_us[index]
            else:
                fba_fee_us = para.fba_fee_big_us + (
                        weight_max_pound - para.limit_weight_big_cal_us) * 16 / 4 * para.fba_fee_big_per_us  # 先转换为盎司再计算
        elif size_tag == para.size_tag_list_us[2]:
            fba_fee_us = para.fba_fee_large_us + (weight_max_pound - 1) * para.fba_fee_large_per_us
        elif size_tag == para.size_tag_list_us[3]:
            fba_fee_us = para.fba_fee_large_50_us + (weight_max_pound - 1) * para.fba_fee_large_50_per_us
        elif size_tag == para.size_tag_list_us[4]:
            fba_fee_us = para.fba_fee_large_70_us + (
                    weight_max_pound - para.limit_weight_large_50_us - 1) * para.fba_fee_large_70_per_us
        elif size_tag == para.size_tag_list_us[5]:
            fba_fee_us = para.fba_fee_large_150_us + (
                    weight_max_pound - para.limit_weight_large_70_us - 1) * para.fba_fee_large_150_per_us
        elif size_tag == para.size_tag_list_us[6]:
            fba_fee_us = para.fba_fee_larger_150_us + (
                    weight_max_pound - para.limit_weight_large_150_us - 1) * para.fba_fee_larger_150_per_us
    else:
        if size_tag == para.size_tag_list_us[0]:
            index = np.digitize([weight_pound], para.weight_small_list_us, right=True)[0] - 1
            fba_fee_us = para.fba_fee_low_small_list_us[index]
        elif size_tag == para.size_tag_list_us[1]:
            if weight_max_pound <= para.limit_weight_big_cal_us:
                index = np.digitize([weight_max_pound], para.weight_big_list_us, right=True)[0] - 1
                fba_fee_us = para.fba_fee_low_big_list_us[index]
            else:
                fba_fee_us = para.fba_fee_low_big_us + (
                        weight_max_pound - para.limit_weight_big_cal_us) * 16 / 4 * para.fba_fee_big_per_us  # 先转换为盎司再计算
        elif size_tag == para.size_tag_list_us[2]:
            fba_fee_us = para.fba_fee_low_large_us + (weight_max_pound - 1) * para.fba_fee_large_per_us
        elif size_tag == para.size_tag_list_us[3]:
            fba_fee_us = para.fba_fee_low_large_50_us + (weight_max_pound - 1) * para.fba_fee_large_50_per_us
        elif size_tag == para.size_tag_list_us[4]:
            fba_fee_us = para.fba_fee_low_large_70_us + (
                    weight_max_pound - para.limit_weight_large_50_us - 1) * para.fba_fee_large_70_per_us
        elif size_tag == para.size_tag_list_us[5]:
            fba_fee_us = para.fba_fee_low_large_150_us + (
                    weight_max_pound - para.limit_weight_large_70_us - 1) * para.fba_fee_large_150_per_us
        elif size_tag == para.size_tag_list_us[6]:
            fba_fee_us = para.fba_fee_low_larger_150_us + (
                    weight_max_pound - para.limit_weight_large_150_us - 1) * para.fba_fee_larger_150_per_us
    return fba_fee_us


# FBA费用计算_UK
def fba_fee_uk_cal(price, size_tag, weight_kg, weight_max_kg):
    fba_fee_uk = 0
    if price <= para.price_low and weight_kg <= para.weight_low_small_bag_max_ukde:
        if size_tag == para.size_tag_list_ukde[0]:
            fba_fee_uk = para.fba_fee_low_small_envelope_uk
        elif size_tag == para.size_tag_list_ukde[1]:
            index = np.digitize([weight_kg], para.weight_standard_envelope_list_ukde, right=True)[0] - 1
            fba_fee_uk = para.fba_fee_low_standard_envelope_list_uk[index]
        elif size_tag == para.size_tag_list_ukde[2]:
            fba_fee_uk = para.fba_fee_low_big_envelope_uk
        elif size_tag == para.size_tag_list_ukde[3]:
            fba_fee_uk = para.fba_fee_low_large_envelope_uk
        elif size_tag == para.size_tag_list_ukde[4]:
            index = np.digitize([weight_kg], para.weight_low_small_bag_list_ukde, right=True)[0] - 1
            fba_fee_uk = para.fba_fee_low_small_bag_list_uk[index]
    else:
        if size_tag == para.size_tag_list_ukde[0]:
            fba_fee_uk = para.fba_fee_small_envelope_uk
        elif size_tag == para.size_tag_list_ukde[1]:
            index = np.digitize([weight_kg], para.weight_standard_envelope_list_ukde, right=True)[0] - 1
            fba_fee_uk = para.fba_fee_standard_envelope_list_uk[index]
        elif size_tag == para.size_tag_list_ukde[2]:
            fba_fee_uk = para.fba_fee_big_envelope_uk
        elif size_tag == para.size_tag_list_ukde[3]:
            fba_fee_uk = para.fba_fee_large_envelope_uk
        elif size_tag == para.size_tag_list_ukde[4]:
            index = np.digitize([weight_max_kg], para.weight_small_bag_list_ukde, right=True)[0] - 1
            fba_fee_uk = para.fba_fee_small_bag_list_uk[index]
        elif size_tag == para.size_tag_list_ukde[5]:
            index = np.digitize([weight_max_kg], para.weight_standard_bag_list_ukde, right=True)[0] - 1
            fba_fee_uk = para.fba_fee_standard_bag_list_uk[index]
        elif size_tag == para.size_tag_list_ukde[6]:
            if weight_max_kg <= para.limit_weight_small_ukde:
                index = np.digitize([weight_max_kg], para.weight_small_list_ukde, right=True)[0] - 1
                fba_fee_uk = para.fba_fee_small_list_uk[index]
            else:
                fba_fee_uk = para.fba_fee_small_uk + (
                        weight_max_kg - para.limit_weight_small_ukde) * para.fba_fee_per_ukde
        elif size_tag == para.size_tag_list_ukde[7]:
            if weight_max_kg <= para.limit_weight_standard_ukde:
                index = np.digitize([weight_max_kg], para.weight_standard_list_ukde, right=True)[0] - 1
                fba_fee_uk = para.fba_fee_standard_list_uk[index]
            else:
                fba_fee_uk = para.fba_fee_standard_uk + (
                        weight_max_kg - para.limit_weight_standard_ukde) * para.fba_fee_per_ukde
        elif size_tag == para.size_tag_list_ukde[8]:
            if weight_max_kg <= para.limit_weight_big_ukde:
                index = np.digitize([weight_max_kg], para.weight_big_list_ukde, right=True)[0] - 1
                fba_fee_uk = para.fba_fee_big_list_uk[index]
            else:
                fba_fee_uk = para.fba_fee_big_uk + (
                        weight_max_kg - para.limit_weight_big_ukde) * para.fba_fee_per_ukde
        elif size_tag == para.size_tag_list_ukde[9]:
            if weight_max_kg <= para.limit_weight_large_ukde:
                index = np.digitize([weight_max_kg], para.weight_large_list_ukde, right=True)[0] - 1
                fba_fee_uk = para.fba_fee_large_list_uk[index]
            else:
                fba_fee_uk = para.fba_fee_large_uk + (
                        weight_max_kg - para.limit_weight_large_ukde) * para.fba_fee_large_per_uk
    return fba_fee_uk


# FBA费用计算_DE
def fba_fee_de_cal(price, size_tag, weight_kg, weight_max_kg):
    fba_fee_de = 0
    if price <= para.price_low_de and weight_kg <= para.weight_low_small_bag_max_ukde:
        if size_tag == para.size_tag_list_ukde[0]:
            fba_fee_de = para.fba_fee_low_small_envelope_de
        elif size_tag == para.size_tag_list_ukde[1]:
            index = np.digitize([weight_kg], para.weight_standard_envelope_list_ukde, right=True)[0] - 1
            fba_fee_de = para.fba_fee_low_standard_envelope_list_de[index]
        elif size_tag == para.size_tag_list_ukde[2]:
            fba_fee_de = para.fba_fee_low_big_envelope_de
        elif size_tag == para.size_tag_list_ukde[3]:
            fba_fee_de = para.fba_fee_low_large_envelope_de
        elif size_tag == para.size_tag_list_ukde[4] and weight_kg <= para.weight_low_small_bag_max_ukde:
            index = np.digitize([weight_kg], para.weight_low_small_bag_list_ukde, right=True)[0] - 1
            fba_fee_de = para.fba_fee_low_small_bag_list_de[index]
    else:
        if size_tag == para.size_tag_list_ukde[0]:
            fba_fee_de = para.fba_fee_small_envelope_de
        elif size_tag == para.size_tag_list_ukde[1]:
            index = np.digitize([weight_kg], para.weight_standard_envelope_list_ukde, right=True)[0] - 1
            fba_fee_de = para.fba_fee_standard_envelope_list_de[index]
        elif size_tag == para.size_tag_list_ukde[2]:
            fba_fee_de = para.fba_fee_big_envelope_de
        elif size_tag == para.size_tag_list_ukde[3]:
            fba_fee_de = para.fba_fee_large_envelope_de
        elif size_tag == para.size_tag_list_ukde[4]:
            index = np.digitize([weight_max_kg], para.weight_small_bag_list_ukde, right=True)[0] - 1
            fba_fee_de = para.fba_fee_small_bag_list_de[index]
        elif size_tag == para.size_tag_list_ukde[5]:
            index = np.digitize([weight_max_kg], para.weight_standard_bag_list_ukde, right=True)[0] - 1
            fba_fee_de = para.fba_fee_standard_bag_list_de[index]
        elif size_tag == para.size_tag_list_ukde[6]:
            if weight_max_kg <= para.limit_weight_small_ukde:
                index = np.digitize([weight_max_kg], para.weight_small_list_ukde, right=True)[0] - 1
                fba_fee_de = para.fba_fee_small_list_de[index]
            else:
                fba_fee_de = para.fba_fee_small_de + (
                        weight_max_kg - para.limit_weight_small_ukde) * para.fba_fee_per_ukde
        elif size_tag == para.size_tag_list_ukde[7]:
            if weight_max_kg <= para.limit_weight_standard_ukde:
                index = np.digitize([weight_max_kg], para.weight_standard_list_ukde, right=True)[0] - 1
                fba_fee_de = para.fba_fee_standard_list_de[index]
            else:
                fba_fee_de = para.fba_fee_standard_de + (
                        weight_max_kg - para.limit_weight_standard_ukde) * para.fba_fee_per_ukde
        elif size_tag == para.size_tag_list_ukde[8]:
            if weight_max_kg <= para.limit_weight_big_ukde:
                index = np.digitize([weight_max_kg], para.weight_big_list_ukde, right=True)[0] - 1
                fba_fee_de = para.fba_fee_big_list_de[index]
            else:
                fba_fee_de = para.fba_fee_big_de + (weight_max_kg - para.limit_weight_big_ukde) * para.fba_fee_per_ukde
        elif size_tag == para.size_tag_list_ukde[9]:
            if weight_max_kg <= para.limit_weight_large_ukde:
                index = np.digitize([weight_max_kg], para.weight_large_list_ukde, right=True)[0] - 1
                fba_fee_de = para.fba_fee_large_list_de[index]
            else:
                fba_fee_de = para.fba_fee_large_de + (
                        weight_max_kg - para.limit_weight_large_ukde) * para.fba_fee_large_per_de
    return fba_fee_de


def profit_cal(price, product_fee, freight_fee, freight_fee_air, fba_fee, vat):
    # 货值计算
    product_fee_rate = product_fee / price
    # 头程
    freight_fee_rate = freight_fee / price
    freight_fee_air_rate = freight_fee_air / price
    # FBA
    fba_fee_rate = fba_fee / price
    # 利润计算：定价 - （FBA物流费用+采购成本+头程运费+交易佣金+汇率折损+退款费用+VAT税）
    profit_rate = 1 - (product_fee_rate + freight_fee_rate + fba_fee_rate + para.referral_fee_rate
                       + para.exchange_loss_rate + para.return_rate + vat)
    profit = price * profit_rate
    # 空运
    profit_air_rate = 1 - (product_fee_rate + freight_fee_air_rate + fba_fee_rate + para.referral_fee_rate
                           + para.exchange_loss_rate + para.return_rate + vat)
    profit_air = price * profit_rate
    return product_fee_rate, freight_fee_rate, freight_fee_air_rate, fba_fee_rate, profit_rate, profit, \
           profit_air_rate, profit_air
