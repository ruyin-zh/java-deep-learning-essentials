package org.ruyin.deep.learning.code.base.util;

import java.util.Random;

/**
 * @author: hjxz
 * @date: 2021/3/22
 * @desc: 高斯分布模型
 *
 */
public final class GaussianDistribution {

    /**
     *
     * 均值
     *
     * */
    private final double mean;
    /**
     *
     * 方差
     *
     * */
    private final double var;
    /**
     *
     * 随机变量
     *
     * */
    private final Random rng;


    public GaussianDistribution(double mean, double var, Random rng) {
        if (var < 0.0){
            throw new IllegalArgumentException("方差不可为负数");
        }

        this.mean = mean;
        this.var = var;

        if (rng == null){
            rng = new Random();
        }
        this.rng = rng;
    }


    public double random(){
        double r = 0.0;
        while (r == 0.0){
            r = rng.nextDouble();
        }

        double c = Math.sqrt(-2.0 * Math.log(r));

        if (rng.nextDouble() < 0.5){
            return c * Math.sin(2.0 *Math.PI * rng.nextDouble()) * var + mean;
        }

        return c * Math.cos(2.0 * Math.PI * rng.nextDouble()) * var + mean;
    }

}
