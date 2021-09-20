1. Add TFText tokenizer support with encode and decode.
2. Edit T5 docs.
3. Add MLM as in BigBird.
4. Add MLM as in tftext.
5. Add CausalLM as in tftext and Megatron.
6. Test ViT.
7. Add multi GPU inference.
8. Add jupytext.
9. Make use_mlm_layer strictly optional.
10. Fix model=encoder in TransformerBERT and TransformerBART
11. Check validate tf.function by moving the metric update inside tf.function.
    Check the effect of strategy.run(tf.reduce_op.SUM)
12. Add gradient clipping.
