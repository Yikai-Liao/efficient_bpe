# 初次门控编译问题

首次冻结版 lib SHA `389417bf36408ca091ff2032f55ae5398788f12c695d2e10a032d931e2529e22`。
`cargo fmt --check` 通过；`cargo test --offline --locked --lib` 在
`src/lib.rs:399:26` 因 E0689 停止：对未明确整数类型的累加器调用
`.checked_add(end - first)`。这是测试/源码编译门槛，未运行 Rust 测试、
oracle 或性能。已把位置交作者限于本 crate 修正；后续以新 SHA
重新构建与门控，本记录保留原始失败来源。
