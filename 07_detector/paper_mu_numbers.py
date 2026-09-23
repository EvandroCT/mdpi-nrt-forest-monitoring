# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""MU numbering of the paper's text and tables (Table S1): MU_1..MU_20 in order of increasing
out-of-fold MSE of the published fold models, keyed by the originating DETER alert identifier
that names each monitoring unit's zarr store. Used only to label table rows."""
MU_NUMBER = {
    "11302_curr": 1, "11348_curr": 2, "12035_curr": 3, "10925_curr": 4, "11864_curr": 5,
    "11303_curr": 6, "11301_curr": 7, "11325_curr": 8, "11304_curr": 9, "13463_curr": 10,
    "11989_curr": 11, "11873_curr": 12, "10923_curr": 13, "12036_curr": 14, "11824_curr": 15,
    "11883_curr": 16, "11866_curr": 17, "12024_curr": 18, "11906_curr": 19, "11322_curr": 20,
}
