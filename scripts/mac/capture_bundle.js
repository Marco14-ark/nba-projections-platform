// Run in the desktop app's built-in browser on a https://www.nba.com page (same-origin headers).
// Produces one JSON download: schedule, latest official injury PDF, pregame boxscores (<= 2.5h to tip),
// and final boxscore + play-by-play for every finished game not yet archived (pass known ids).
// Gentle pacing; on any non-200 the item is recorded as missing — nothing is invented.
async function captureBundle(knownFinalIds = [], fname = "nba_capture.json") {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const now = new Date();
  const sched = await fetch("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json").then(r => r.json());
  const games = [];
  for (const d of sched.leagueSchedule.gameDates) for (const g of d.games) if (g.gameId.startsWith("002"))
    games.push({gameId: g.gameId, t: g.gameDateTimeUTC, h: g.homeTeam.teamId, a: g.awayTeam.teamId,
                ht: g.homeTeam.teamTricode, at: g.awayTeam.teamTricode, st: g.gameStatus});
  // latest injury report: walk back 15-minute slots (ET)
  let pdf = null, url = null;
  const et = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
  for (let k = 0; k < 20 && !pdf; k++) {
    const t = new Date(et.getTime() - k * 15 * 60000); t.setMinutes(Math.floor(t.getMinutes() / 15) * 15);
    const h12 = (t.getHours() % 12) || 12, ap = t.getHours() >= 12 ? "PM" : "AM";
    const stamp = `${t.getFullYear()}-${String(t.getMonth()+1).padStart(2,"0")}-${String(t.getDate()).padStart(2,"0")}_${String(h12).padStart(2,"0")}_${String(t.getMinutes()).padStart(2,"0")}${ap}`;
    const u = `https://ak-static.cms.nba.com/referee/injury/Injury-Report_${stamp}.pdf`;
    const r = await fetch(u); await sleep(400);
    if (r.status === 200) { const b = new Uint8Array(await r.arrayBuffer()); let s = ""; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); pdf = btoa(s); url = u; }
  }
  const pre = {}, finals = [], missing = [], live = {};
  for (const g of games) {
    const dt = (new Date(g.t) - now) / 3600000;
    if (g.st === 1 && dt > 0 && dt <= 2.5) {
      const r = await fetch(`https://cdn.nba.com/static/json/liveData/boxscore/boxscore_${g.gameId}.json`); await sleep(300);
      pre[g.gameId] = r.status === 200 ? (await r.json()).game : "NOT_POSTED";
    }
    if (g.st === 2) {   // in progress: score only, timestamped; never used as model input
      const r = await fetch(`https://cdn.nba.com/static/json/liveData/boxscore/boxscore_${g.gameId}.json`); await sleep(300);
      if (r.status === 200) { const b = (await r.json()).game;
        live[g.gameId] = {status: b.gameStatusText, period: b.period, clock: b.gameClock, home: b.homeTeam.score, away: b.awayTeam.score, captured_at: new Date().toISOString()}; }
    }
    if (g.st === 3 && !knownFinalIds.includes(g.gameId)) {
      const b = await fetch(`https://cdn.nba.com/static/json/liveData/boxscore/boxscore_${g.gameId}.json`); await sleep(300);
      const p = await fetch(`https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_${g.gameId}.json`); await sleep(300);
      if (b.status === 200 && p.status === 200) finals.push({gameId: g.gameId, actions: (await p.json()).game.actions, box: (await b.json()).game});
      else missing.push(g.gameId);
    }
  }
  // official team rosters (current teams; players without box appearances yet)
  const rosters = [];
  const teamIds = [...new Set(games.flatMap(g => [g.h, g.a]))].filter(t => t > 0);
  for (const tid of teamIds) {
    try {
      const r = await fetch(`https://stats.nba.com/stats/commonteamroster?LeagueID=00&Season=2026-27&TeamID=${tid}`); await sleep(400);
      if (r.status === 200) { const j = await r.json(); const rs = j.resultSets[0]; const H = rs.headers;
        for (const row of rs.rowSet) { const o = {}; H.forEach((h, i) => o[h] = row[i]); rosters.push({TeamID: o.TeamID, PLAYER_ID: o.PLAYER_ID, PLAYER: o.PLAYER, POSITION: o.POSITION}); } }
    } catch (e) {}
  }
  const out = {captured_at: now.toISOString(), rosters, injury_pdf_url: url, injury_pdf_b64: pdf, schedule: games, pregame_box: pre, finals, missing_finals: missing, live};
  const blob = new Blob([JSON.stringify(out)], {type: "application/json"}); const el = document.createElement("a");
  el.href = URL.createObjectURL(blob); el.download = fname; document.body.appendChild(el); el.click();
  return {games: games.length, pdf: url, pregame: Object.keys(pre).length, finals: finals.length, missing, bytes: blob.size};
}
