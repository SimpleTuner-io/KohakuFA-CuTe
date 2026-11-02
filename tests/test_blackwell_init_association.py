import unittest

from benchmarks.diagnostics.blackwell_init_association import associate, initcheck_counts


class InitAssociationTests(unittest.TestCase):
    log = """========= Uninitialized __shared__ memory read of size 4 bytes
=========     at allocation_probe+0xee0
=========     by thread (384,0,0) in block (0,0,0)
=========     Address 0x54
Terminated
"""
    sass = """Function : allocation_probe
/*0df0*/ UMOV UR12, 0x40 ;
/*0e10*/ UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR8, UR8 ;
/*0ee0*/ ATOMS.OR RZ, [UR12+0x14], R3 ;
Function : unrelated
/*0ee0*/ LDS R2, [UR17+0x188] ;
"""

    def test_atomic_report_is_not_misidentified_as_pointer_load(self):
        report = associate(self.log, self.sass)
        read = report["reports"][0]
        self.assertTrue(read["candidate_matches_report"])
        self.assertTrue(read["is_shared_atomic_read_modify_write"])
        self.assertEqual(read["nearest_literal_address_candidate"], "0x54")
        self.assertTrue(report["log_was_terminated"])
        self.assertIn("runtime loaded-code binding remains required", report["association"])

    def test_register_overwrite_prevents_address_inference(self):
        sass = self.sass.replace("/*0ee0*/", "/*0e20*/ UIADD3 UR12, UPT, UPT, UR13, URZ, URZ ;\n/*0ee0*/", 1)
        read = associate(self.log, sass)["reports"][0]
        self.assertIsNone(read["nearest_literal_address_candidate"])
        self.assertIsNone(read["candidate_matches_report"])

    def test_repeated_report_and_size_remain_explicit(self):
        report = associate(self.log + self.log, self.sass)
        self.assertEqual(len(report["reports"]), 2)
        self.assertEqual(report["reports"][0]["bytes"], 4)
        self.assertEqual(report["report_classes"][0]["count"], 2)

    def test_missing_pc_symbol_or_duplicate_symbol_is_rejected(self):
        for sass in (self.sass.replace("0ee0", "0ef0"), self.sass.replace("allocation_probe", "other"), self.sass * 2):
            with self.subTest(kind=len(sass)), self.assertRaises(ValueError):
                associate(self.log, sass)

    def test_inline_report_requires_explicit_matching_kernel(self):
        log = self.log.replace("allocation_probe+0xee0", "void inspect<(bool)1>(int *)+0xee0 in probe.cu:75")
        log = log.replace("Terminated", "========= Device Frame: allocation_probe+0x20 in probe.cu:95")
        report = associate(log, self.sass, symbol="allocation_probe")
        self.assertEqual(report["reports"][0]["source_line"], 75)
        self.assertEqual(report["reports"][0]["selected_kernel"], "allocation_probe")
        with self.assertRaises(ValueError):
            associate(log, self.sass)
        with self.assertRaises(ValueError):
            associate(log.replace("Device Frame: allocation_probe", "Device Frame: other"), self.sass, "allocation_probe")

    def test_inline_source_lineage_does_not_add_callsite_pc(self):
        from benchmarks.diagnostics.blackwell_init_association import inline_location

        text = """.section .text.allocation_probe,"ax",@progbits
//## File "probe.cu", line 75 inlined at "probe.cu", line 95
//## File "probe.cu", line 95
/*0190*/ LDS.U8 R3, [R2+0x1c] ;
"""
        self.assertEqual(inline_location(text, "allocation_probe", 0x190), (75, 95))
        self.assertIsNone(inline_location(text, "allocation_probe", 0x1B0))

    def test_truncated_report_does_not_borrow_next_address(self):
        first = self.log.replace("=========     Address 0x54\n", "")
        with self.assertRaises(ValueError):
            associate(first + self.log, self.sass)

    def test_unprinted_subtotal_is_not_a_second_error_total(self):
        text = self.log * 6 + "========= ERROR SUMMARY: 102 errors\n"
        text += "========= ERROR SUMMARY: 96 errors were not printed (print limit reached)\n"
        counts = initcheck_counts(text)
        self.assertEqual(counts["summary_errors"], 102)
        self.assertEqual(counts["unprinted_errors"], 96)
        self.assertFalse(counts["all_errors_are_printed_initialization_reports"])

    def test_all_reports_and_clean_summary_are_distinguished(self):
        counts = initcheck_counts(self.log * 102 + "========= ERROR SUMMARY: 102 errors\n")
        self.assertTrue(counts["all_errors_are_printed_initialization_reports"])
        self.assertEqual(initcheck_counts("========= ERROR SUMMARY: 0 errors\n")["summary_errors"], 0)
        # Other error classes or an interrupted log must not become an all-read classification.
        self.assertFalse(
            initcheck_counts("========= ERROR SUMMARY: 2 errors\n")["all_errors_are_printed_initialization_reports"]
        )
        for log in (self.log, "========= ERROR SUMMARY: 0 errors\n" + self.log):
            with self.assertRaises(ValueError):
                initcheck_counts(log)

    def test_full_allocator_report_has_three_distinct_instruction_classes(self):
        sass = """Function : allocation_probe
/*0150*/ MOV R2, 0x40 ;
/*0180*/ LEA R2, R4, R2, 0x18 ;
/*0190*/ LDS.U8 R3, [R2+0x1c] ;
/*0300*/ ATOMS.OR RZ, [R2+0x14], R5 ;
/*0310*/ ATOMS.OR RZ, [R2+0x18], R7 ;
"""
        reports = []
        for _ in range(3):
            for thread in range(384, 416):
                reports.append(
                    self.log.replace("size 4", "size 1")
                    .replace("0xee0", "0x190")
                    .replace("0x54", "0x5c")
                    .replace("(384,0,0)", f"({thread},0,0)")
                )
            reports.extend((self.log.replace("0xee0", "0x300"), self.log.replace("0xee0", "0x310").replace("0x54", "0x58")))
        log = "".join(reports) + "========= ERROR SUMMARY: 102 errors\n"
        report = associate(log, sass)
        self.assertTrue(initcheck_counts(log)["all_errors_are_printed_initialization_reports"])
        self.assertEqual(
            [(c["pc"], c["count"]) for c in report["report_classes"]], [("0x190", 96), ("0x300", 3), ("0x310", 3)]
        )
        self.assertEqual(report["reports"][0]["thread"], (384, 0, 0))
        self.assertTrue(all(r["block"] == (0, 0, 0) for r in report["reports"]))
        self.assertTrue(all(r["nearest_literal_address_candidate"] is None for r in report["reports"]))

    def test_no_report_is_not_a_clean_result(self):
        with self.assertRaises(ValueError):
            associate("Execution failed before any kernel launched.", self.sass)


if __name__ == "__main__":
    unittest.main()
