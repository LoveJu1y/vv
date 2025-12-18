12.9
最恶心的发现，dropout参数穿错了，nodropout是0，没有的都按1算了。还训个屁，drop out参数还是不要改了我觉得。这是12.9的教训
- bridge_lerobot_fullcot_stage_bt16_lrhalf:
vlm和action dit的lr降为一半，loss无明显变化

- bridge_lerobot_fullcot_stage_bt16_lr1_nodropout
和原始参数几乎没区别，除了把dropout变为0了，但从loss看，0.2的drop和0的dro几乎没区别。

- bridge_lerobot_fullcot_stage_bt16_lr1e-6_nodropout
两者的lr直接降低10倍，loss收敛变慢，学习率太小。

- bridge_lerobot_fullcot_stage_bt16_lr1e-4_ditb_nodropout_bts24
原始参数，只把bts变为24.之前观察到的bts越大，loss方差越小，原因可能单纯是因为step的压缩效应，而不是batch导致的学习率分配问题。从减半学习率可以看到.

受害列表有
- bridge_lerobot_fullcot_stage_bt16_lr1e-6_ditb
- bridge_lerobot_fullcot_stage_bt16_lr1e-4_ditb_bts24

今天的实验当作预警了，必须有计划的跑实验
修改参数：
DATE    DIT  LR_ACTION  LR_VLM  BTS  TOTAL_STEP   SUM_LATNET?   SCORE
12.8
 1       L       5E-5    5E-6    16      60K             N       0.552083(40K)   NO DROPOUT
 2       L       1E-5    1E-6    16      60K             N       0.427083        NO DROPOUT
 3       B       1E-4    1E-5    24      60K             N       0.645833（36K） NO DROPOUT
 4       L       1E-4    1E-5    16      60K             N       0.687500（30K） NO DROPOUT
结论：大batch似乎并没有带来额外收益。测评准确率方差大，集中在55-65，你dropout暂时不考虑，另，学习率折半没有带来收益
12.9
 0       L       1E-4    1E-5    16      60K             N       0.63(40K)     
12.10
 5       B       1E-4    1E-5    16      60K             N       0.697917(35K)
 6       B       1E-4    1E-5    24      60K             N       0.635417(20k&50k)
 7       B       1E-4    1E-5    16      45K             N       0.625000(45K)
 8       B       5E-5    5E-6    24      45K             N       0.593750(35K)
 9       L       1E-4    1E-5    16      45K             N       0.635417(35K)
10       L       5E-5    5E-6    16      45K             N       0.45(LOW)
初步的结论是使用DITB足以（0和5，但还不足以作证），同时NO DROPOUT似乎带来了一些收益（0，4），它能让模型拟合加速，学习率保持原来的样子即可。另一个问题是，似乎，45K的step没有究竟有没有必要，需要另一些实验来验证

总结的话就是：
1. 学习率不用调整了，Dropout或许值得再验证一下，后续设置0.1的drop实验看一下。使用DITB！均值更高，最大值也更高
2. （5和6，可以对比一下batchsize有没有作用，5和7可以对比一下训练step有没有用。）
3. 得到这些结论初步实验就够了，接下来进行有film或者sum的实验，再验证一些东西。

12.11   
11       B       1E-4    1E-5    16      45K             Y      0.656250（30K）
12       B       1E-4    1E-5    16      60K             Y      <0.6,low
13       L       1E-4    1E-5    16      45K             Y      0.645833(45K)
14       L       1E-4    1E-5    16      60K             Y      <0.6,更low

12.12
试验一下dropout为0.1吧。
12       B       1E-4    1E-5    16      60K             Y      0.645833（45K，dr0.1）


今天跑一下上面的实验，下一阶段应该做？目前对于上述参数的调节先这样吧？作为初步结论，因为模型还得改：
1. action token加上，vlm loss weight 0.1试试？
2. 在DIT中加入film，或者直接考虑latent token作为state feature
bash -i -c "
  unset http_proxy && unset https_proxy &&
  conda activate starVLA &&
  cd /share/project/lvjing/starVLA/ &&
  MAX_TRAIN_STEPS=45000 \
  PER_DEVICE_BATCH=16 \
  LR_VLM=1e-5 \
  LR_ACTION=1e-4 \
  ACTION_DIT_TYPE=DiT-L \
  RUN_ID=bridge_lerobot_DITL_LR1E-4_LR1E-5_BTS16_45K \
  bash /share/project/lvjing/starVLA/scripts/run_starvla_bridge.sh
"

现在有点不知道干什么了，只能不断迭代加测评了，训出来就算成功。
上述任务跑完再开始做1 2 选项。明天晚上之前吧！

12.11：
我们直接用更新后的网络训练吧，前期实验当作一个参数调节的基础先验，探究清楚学习率不需要对半，同时步数维持在50k时可以，绝大部分在45K前取得极大。

现在已经跑起来的eval的有：
results/BridgeFinal_Action/bridge_lerobot_DITL_LR5E-5_LR5E-6_BTS16_45K  yes
results/BridgeFinal_Action/bridge_lerobot_DITL_LR1E-4_LR1E-5_BTS16_45K  yes

results/BridgeFinal_Action/bridge_lerobot_DITB_LR5E-5_LR5E-6_BTS16_45K  yes
results/BridgeFinal_Action/bridge_lerobot_DITB_LR1E-4_LR1E-5_BTS16_45K  no

这两个用来比较学习率的影响，但从之前的经验来看，学习率不用调节。
结论：不用调节！


results/BridgeFinal_Action/bridge_lerobot_DITB_LR1E-4_LR1E-5_BTS24_60K  no
results/BridgeFinal_Action/bridge_lerobot_DITB_LR1E-4_LR1E-5_BTS16_60K  yes
用来比较batchsize是否有影响，但是这两个学习率都低，可以参考一下
结论：待定


12.12：
现在有一个这样的想法：
第一阶段完全cot10k，我觉得是ok的，我想着按这样的顺序，比如，先subtask，再move reasoning，再bbox，因为我觉得bbox是重要的，或者bbox放在中间吧，毕竟训练数据是有限的。可以我觉得，重新训三个阶段。然后最后一个lantent阶段，DIT直接开着，sum或者film也开着？也是训个10k？我觉得是合理的，最后一个阶段loss如何分配呢？给vlm loss确实加一个0.2的loss weight吧。ok，那就按照新的setting，修改一下训练吧

更新：新的vlm跑起来了，规划一下下一步怎么跑训练呢？

12.13：
跑一下drop=0.1的，确定就用ditb吧，太大了也没什么用。

我觉得基于新训的vlm2，直接45k，加入sum吧，先试试，

确定用ditb了。然后试一下summary=1
事实证明，drop=0.1吧，没有不好看，0.1明显有提升

基于以上发现，今天跑以下实验吧：
- 首先我们已经确定，DITB，学习率不变，dr=0.1，目前60k在drop=0.1学的比0.2好这么多，很奇怪，
- 有了vlm2，我们基于vlm2，跑以下实验：
  基于vlm2的，dit b，1e-4,1e-5,45k/60 k,no sum
  基于vlm2的，dit b，1e-4,1e-5,45/60k ,sum=2,
  基于vlm2的，dit b，1e-4,1e-5,45/60k,sum=2, dr=0.1交叉验证一下
  同时需要跑一个基于vlm1的 ditb 1e-4,1e-5,45k ,sum=2,dr=0.1

  七个实验！。先清理一波存储！

12.17
我觉得还是需要把这个当作一个事干，认真好吧，你必须认真思考一下，现在需要做哪些实验？我需要对比哪些方面？哪些应该是必须做的，哪些是随后需要验证的？

我的变量现在是有两个的，一个是cot的数据，一个是隐式推理，我需要证明我这两部分都有贡献，因而，我需要做如下的实验：
- 首先，用原始的无cot数据，和有cot的对比。
- 其次，用有cot的，对比用了隐式推理的。
在上述计划中，我们显然是需要记录时间和准确率两个的。
我正在犹豫，要不要使用groot结构做上面的实验。如果不做，展示出来的更多作用就在于groot结构了而非我的其它贡献。换句话说，作为消融实验，上面的结论确实可以证明想要消融的结论，但是，假如说，我第一个那样消融没问题，第二个，肯定得用我的自己的最终模型了，因为如果不用，跑出来60，那不就说明我最终的核心提点是groot结构吗，对吧。所以最合理的还是需要一切都基于groot结构去做。

没问题，那么，在这个过程中，我们依然需要统一口径，然后重点是，在代码方面，我需要完成以下几个任务，
首先，无cot的训练，这个是最好实现的了，只需要修改dataformatter，把stage0标记为无cot即可。测评的时候，什么都不需要改，一遍forward，直接出action就行。
第二个，有cot的怎么实现呢？全cot训10k，然后再45k groot。测评的时候主要得出一个自回归生成cot的步骤，也就是generate部分。
第三个，就是film的消融实验，不需要修改代码几乎。

所以所有实验都可以基于groot去做。那么，我们只能祈祷了。这个计划我觉得是可以的。

这会先跑一个cot0阶段吧！
  