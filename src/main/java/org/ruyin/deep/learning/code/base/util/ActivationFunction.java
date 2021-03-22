package org.ruyin.deep.learning.code.base.util;

/**
 * @author: hjxz
 * @date: 2021/3/22
 * @desc: 激活函数
 *
 */
public final class ActivationFunction {


    public static int step(double x){
        if (x >= 0){
            return 1;
        }else {
            return -1;
        }
    }

}
