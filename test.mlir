module {
  func.func @index(%0: tensor<5xi32>) -> i64 {
    %1 = arith.constant 0: i64
    %2 = "index.castu"(%1) : (i64) -> index
    return %1 : i64
  }
}
