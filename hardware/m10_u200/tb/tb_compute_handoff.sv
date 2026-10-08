`timescale 1ns/1ps
/** Registered ready-stage fast paths, stalled payloads, and clear collisions. */
module tb_compute_handoff;
  import tile_prefetch_package::*;
  logic clk = 0;
  always #2.5 clk = ~clk;
  logic rst_n = 0;
  logic clear = 0;
  logic request_valid = 0;
  logic request_ready;
  logic gu_resident = 0;
  logic down_resident = 0;
  logic gu_done = 0;
  logic nonlinear_done = 0;
  logic down_done = 0;
  logic busy, done, waiting_gu, waiting_down, gu_start, down_start, release_valid;
  logic gu_release;
  logic [15:0] release_gu, release_down, release_epoch;
  logic [7:0] release_gu_generation, release_down_generation;
  tile_descriptor_t gu, down;

  tile_compute_controller #(
    .EXTERNAL_COMPUTE(1'b1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .compute_req_valid(request_valid),
    .compute_req_ready(request_ready),
    .compute_gu_descriptor(gu),
    .compute_gu_slot_generation(8'd5),
    .compute_down_descriptor(down),
    .compute_down_slot_generation(8'd9),
    .gu_resident(gu_resident),
    .down_resident(down_resident),
    .gu_cycles(32'd1),
    .nonlinear_cycles(32'd1),
    .down_cycles(32'd1),
    .backend_gu_done(gu_done),
    .backend_nonlinear_done(nonlinear_done),
    .backend_down_done(down_done),
    .busy(busy),
    .done(done),
    .waiting_gu(waiting_gu),
    .waiting_down(waiting_down),
    .gu_compute_start(gu_start),
    .down_compute_start(down_start),
    .consumer_gu_release_valid(gu_release),
    .consumer_release_valid(release_valid),
    .consumer_release_gu_slot(release_gu),
    .consumer_release_down_slot(release_down),
    .consumer_release_epoch(release_epoch),
    .consumer_release_gu_generation(release_gu_generation),
    .consumer_release_down_generation(release_down_generation)
  );

  task automatic tick;
    @(posedge clk);
    #0.1;
  endtask

  task automatic accept_ready;
    @(negedge clk);
    gu_resident = 1;
    down_resident = 1;
    request_valid = 1;
    tick();
    if (!gu_start || !busy || request_ready) $fatal(1,"ready GU did not start at acceptance edge");
    @(negedge clk);
    request_valid = 0;
  endtask

  initial begin
    gu = '0;
    gu.epoch_id = 7;
    gu.ocm_slot = 2;
    gu.matrix_kind = MATRIX_GU;
    down = gu;
    down.ocm_slot = 3;
    down.matrix_kind = MATRIX_DOWN;
    repeat (3) @(negedge clk);
    rst_n = 1;
    for (integer mode = 0; mode < 4; mode = mode + 1) begin
      @(negedge clk);
      gu_resident = !(mode & 1);
      down_resident = !(mode & 2);
      request_valid = 1;
      tick();
      if (!busy || request_ready || gu_start != gu_resident || down_start || release_valid)
        $fatal(1,"GU acceptance fast-path contract failed mode=%0d",mode);
      if (release_gu != 2 || release_down != 3 || release_epoch != 7 ||
          release_gu_generation != 5 || release_down_generation != 9)
        $fatal(1,"acceptance did not capture descriptor/generation before start");
      @(negedge clk);
      request_valid = 0;
      if (mode & 1) begin
        repeat (3) begin
          tick();
          if (!waiting_gu || gu_start || down_start || release_valid) $fatal(1,"unready GU advanced");
        end
        @(negedge clk);
        gu_resident = 1;
        tick();
        if (!gu_start) $fatal(1,"delayed GU did not start on readiness edge");
      end
      repeat (3) begin
        tick();
        if (gu_start || down_start || done || release_valid || !busy)
          $fatal(1,"external stage advanced using timer or repeated start");
      end
      @(negedge clk);
      gu_done = 1;
      tick();
      if (!gu_release || down_start || done || release_valid) $fatal(1,"GU release or nonlinear dependency failed");
      @(negedge clk);
      gu_done = 0;
      tick();
      @(negedge clk);
      nonlinear_done = 1;
      tick();
      if (down_start != down_resident || done || release_valid)
        $fatal(1,"ready DOWN did not launch at nonlinear completion edge");
      @(negedge clk);
      nonlinear_done = 0;
      if (mode & 2) begin
        repeat (3) begin
          tick();
          if (!waiting_down || down_start || release_valid || done) $fatal(1,"unready DOWN advanced");
        end
        @(negedge clk);
        down_resident = 1;
        tick();
        if (!down_start) $fatal(1,"delayed DOWN did not start on readiness edge");
      end
      repeat (3) begin
        tick();
        if (down_start || done || release_valid || request_ready) $fatal(1,"DOWN repeated or retired early");
      end
      @(negedge clk);
      down_done = 1;
      tick();
      if (!done || !release_valid || busy || !request_ready) $fatal(1,"retirement failed");
      @(negedge clk);
      down_done = 0;
      tick();
      if (done || release_valid) $fatal(1,"retirement repeated");
    end
    // Clear wins over a would-be ready GU acceptance.
    @(negedge clk);
    clear = 1;
    request_valid = 1;
    gu_resident = 1;
    tick();
    if (gu_start || down_start || gu_release || busy || done || release_valid || request_ready)
      $fatal(1,"clear/acceptance collision leaked work");
    @(negedge clk);
    clear = 0;
    request_valid = 0;
    accept_ready();
    // Clear wins over GU completion and its new independent release pulse.
    @(negedge clk);
    gu_done = 1;
    clear = 1;
    tick();
    if (gu_release || down_start || gu_start || busy || done || release_valid)
      $fatal(1,"clear/GU completion collision leaked release");
    @(negedge clk);
    clear = 0;
    gu_done = 0;
    accept_ready();
    gu_done = 1;
    tick();
    @(negedge clk);
    gu_done = 0;
    nonlinear_done = 1;
    clear = 1;
    tick();
    if (down_start || gu_start || gu_release || busy || done || release_valid)
      $fatal(1,"clear/nonlinear completion collision launched DOWN");
    @(negedge clk);
    clear = 0;
    nonlinear_done = 0;
    accept_ready();
    gu_done = 1;
    tick();
    @(negedge clk);
    gu_done = 0;
    nonlinear_done = 1;
    tick();
    @(negedge clk);
    nonlinear_done = 0;
    down_done = 1;
    clear = 1;
    tick();
    if (done || release_valid || busy) $fatal(1,"clear/retirement collision leaked result");
    @(negedge clk);
    clear = 0;
    down_done = 0;
    repeat (3) begin
      tick();
      if (gu_start || down_start || done || release_valid || busy) $fatal(1,"flush left stale event");
    end
    $display("PASS: registered GU/DOWN fast paths, 4 readiness combinations and clear collisions");
    $finish;
  end
  initial begin
    #10000;
    $fatal(1,"compute handoff watchdog");
  end
endmodule
