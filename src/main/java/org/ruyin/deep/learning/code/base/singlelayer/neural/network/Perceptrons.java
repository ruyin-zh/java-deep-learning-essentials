package org.ruyin.deep.learning.code.base.singlelayer.neural.network;

import org.ruyin.deep.learning.code.base.util.ActivationFunction;
import org.ruyin.deep.learning.code.base.util.GaussianDistribution;

import java.util.Random;

/**
 * @author: hjxz
 * @date: 2021/3/22
 * @desc: 单层神经网络感知器
 *
 */
public class Perceptrons {

    /**
     *
     * 输入数据维度
     *
     * */
    private int nIn;

    /**
     *
     * 维度数据的权重
     *
     * */
    private double[] w;

    public Perceptrons(int nIn) {
        this.nIn = nIn;
        w = new double[nIn];
    }

    /**
     *
     * 采用梯度下降法对数据进行训练
     * @param train_x 训练数据对应的变量
     * @param train_t 训练数据对应的值
     * @param learningRate 学习率
     *
     * */
    public int train(double[] train_x, int train_t, double learningRate){

        int classified = 0;
        double c = 0.;

        //校验数据是否被正确分类
        for (int i = 0; i < nIn; i++){
            c += w[i] * train_x[i] * train_t;
        }

        //如果数据分类错误,则使用梯度下降法对权重进行调整
        if (c > 0){
            classified = 1;
        }else {
            for (int i = 0; i < nIn; i++){
                w[i] += learningRate * train_x[i] * train_t;
            }
        }

        return classified;
    }


    public int predict(double[] x){
        double preActivation = 0.;

        for (int i = 0; i < nIn; i++){
            preActivation += w[i] * x[i];
        }

        return ActivationFunction.step(preActivation);
    }


    public static void main(String[] args) {
        //训练数据数据量
        final int train_N = 1000;
        //测试数据的数据量
        final int test_N = 200;
        //输入数据的维度
        final int nIn = 2;

        //用于训练的输入数据
        double[][] train_X = new double[train_N][nIn];
        //用于训练的输出结果
        int[] train_T = new int[train_N];

        //用于测试的输入数据
        double[][] test_X = new double[test_N][nIn];
        //用于测试的数据的实际标记
        int[] test_T = new int[test_N];
        //模型预测的输出数据
        int[] predicted_T = new int[test_N];

        //最大迭代次数
        final int epochs = 2000;
        //感知器中学习率可以为1
        final double learningRate = 1.;


        //创建训练数据集集测试数据集
        final Random rng = new Random(1234);
        GaussianDistribution g1 = new GaussianDistribution(-2.0,1.0,rng);
        GaussianDistribution g2 = new GaussianDistribution(2.0,1.0,rng);

        //第一类数据集
        for (int i = 0; i < train_N/2 - 1; i++){
            train_X[i][0] = g1.random();
            train_X[i][1] = g2.random();
            train_T[i] = 1;
        }

        for (int i = 0; i < test_N/2 - 1; i++){
            test_X[i][0] = g1.random();
            test_X[i][1] = g2.random();
            test_T[i] = 1;
        }



        //第二类数据集
        for (int i = train_N / 2; i < train_N; i++){
            train_X[i][0] = g2.random();
            train_X[i][1] = g1.random();
            train_T[i] = -1;
        }

        for (int i = test_N / 2; i < test_N; i++){
            test_X[i][0] = g2.random();
            test_X[i][1] = g1.random();
            test_T[i] = -1;
        }


        int epoch = 0;
        Perceptrons classifier = new Perceptrons(nIn);
        //训练模型
        while (true){
            int classified_ = 0;

            for (int i = 0; i < train_N; i++){
                classified_ += classifier.train(train_X[i],train_T[i],learningRate);
            }

            //当所有的数据都能够被争取的分类则直接停止
            if (classified_ == train_N){
                break;
            }

            epoch++;
            //当训练达到最大迭代次数时直接停止
            if (epoch > epochs){
                break;
            }
        }


        //测试
        for (int i = 0; i < test_N; i ++){
            predicted_T[i] = classifier.predict(test_X[i]);
        }


        //模型评估
        //结果存在四类:TP,TN,FP,FN
        int[][] confusionMatrix = new int[2][2];
        double accuracy = 0.;
        double precision = 0.;
        double recall = 0.;

        for (int i = 0; i < test_N; i++){
            if (predicted_T[i] > 0){
                if (test_T[i] > 0){
                    accuracy += 1;
                    precision += 1;
                    recall += 1;
                    confusionMatrix[0][0] += 1;
                }else {
                    confusionMatrix[1][0] += 1;
                }
            }else {
                if (test_T[i] > 0){
                    confusionMatrix[0][1] += 1;
                }else {
                    accuracy += 1;
                    confusionMatrix[1][1] += 1;
                }
            }
        }

        accuracy /= test_N;
        precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
        recall /= confusionMatrix[0][0] + confusionMatrix[0][1];


        System.out.println("-------------------------------");
        System.out.println("Perceptrons model evaluation");
        System.out.println("-------------------------------");
        System.out.printf("Accuracy:  %.1f %%\n", accuracy * 100);
        System.out.printf("Precision: %.1f %%\n", precision * 100);
        System.out.printf("Recall:    %.1f %%\n", recall * 100);
    }





}
