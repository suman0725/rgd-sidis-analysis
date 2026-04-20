#include "TCanvas.h"
#include "hipo4/RHipoDS.hxx"
#include "ROOT/RVec.hxx"

void hipo_to_root(const char* fname = "rec_clas_.hipo",
                   const char* fout  = "output.root") {

  ROOT::DisableImplicitMT();
  auto df = MakeHipoDataFrame(fname);

  // ── columns to keep ──────────────────────────────────────────────
  std::vector<std::string> keep = {

    // run identification
    "RUN_config_run",
    "RUN_config_event",
    "RUN_config_torus",
    "RUN_config_solenoid",

    // event-level
    "REC_Event_helicity",
    "REC_Event_helicityRaw",
    "REC_Event_startTime",
    "REC_Event_RFTime",

    // reconstructed particle (core)
    "REC_Particle_pid",
    "REC_Particle_px",
    "REC_Particle_py",
    "REC_Particle_pz",
    "REC_Particle_vx",
    "REC_Particle_vy",
    "REC_Particle_vz",
    "REC_Particle_charge",
    "REC_Particle_beta",
    "REC_Particle_chi2pid",
    "REC_Particle_status",
    "REC_Particle_vt",

    // calorimeter (PCAL + EC for electron SF and fiducial)
    "REC_Calorimeter_pindex",
    "REC_Calorimeter_sector",
    "REC_Calorimeter_layer",
    "REC_Calorimeter_energy",
    "REC_Calorimeter_x",
    "REC_Calorimeter_y",
    "REC_Calorimeter_z",
    "REC_Calorimeter_lu",
    "REC_Calorimeter_lv",
    "REC_Calorimeter_lw",

    // cherenkov (HTCC for electron ID)
    "REC_Cherenkov_pindex",
    "REC_Cherenkov_detector",
    "REC_Cherenkov_sector",
    "REC_Cherenkov_nphe",

    // scintillator (FTOF for pion beta/timing)
    "REC_Scintillator_pindex",
    "REC_Scintillator_detector",
    "REC_Scintillator_sector",
    "REC_Scintillator_layer",
    "REC_Scintillator_component",
    "REC_Scintillator_time",
    "REC_Scintillator_path",

    // track (FD sector, chi2 for quality)
    "REC_Track_pindex",
    "REC_Track_detector",
    "REC_Track_sector",
    "REC_Track_status",
    "REC_Track_chi2",
    "REC_Track_NDF",

    // trajectory (DC fiducial cuts — region 1,2,3 x,y,z)
    "REC_Traj_pindex",
    "REC_Traj_detector",
    "REC_Traj_layer",
    "REC_Traj_x",
    "REC_Traj_y",
    "REC_Traj_z",
    "REC_Traj_edge",
  };

  // ── safety check: only keep columns that actually exist ──────────
  auto available = df.GetColumnNames();
  std::vector<std::string> final_cols;
  for (const auto& col : keep) {
    if (std::find(available.begin(), available.end(), col) != available.end()) {
      final_cols.push_back(col);
    } else {
      std::cout << "[WARN] Column not found, skipping: " << col << "\n";
    }
  }

  // ── write only the selected columns ─────────────────────────────
  ROOT::RDF::RSnapshotOptions opts;
  opts.fMode        = "RECREATE";
  opts.fCompressionLevel = 4;

  df.Snapshot("data", fout, final_cols, opts);

  std::cout << "[DONE] Wrote " << final_cols.size()
            << " columns to " << fout << "\n";
}